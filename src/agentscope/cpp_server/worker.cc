#include "worker.h"

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/sem.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <random>

using std::getenv;
using std::to_string;

using namespace pybind11::literals;

using WorkerArgs::AgentArgs;
using WorkerArgs::AgentFuncArgs;
using WorkerArgs::AgentFuncReturn;
using WorkerArgs::AgentListReturn;
using WorkerArgs::CreateAgentArgs;
using WorkerArgs::ModelConfigsArgs;
using WorkerArgs::MsgReturn;

#define LOG(...) RAW_LOGGER(this, __VA_ARGS__)
#define ERROR_MSG_FORMAT(msg) \
  (string("Error: ") + msg + "failed in " + __FUNCTION__)
#define ERROR(msg)            \
  LOG(ERROR_MSG_FORMAT(msg)); \
  perror(ERROR_MSG_FORMAT(msg).c_str())

#define PY_LOG(level, content)       \
  if (_enable_py_logger) {           \
    _py_logger.attr(level)(content); \
  }

inline int P(int semid, unsigned short sem_num, short sem_flg = 0) {
  struct sembuf sb = {sem_num, -1, sem_flg};
  return semop(semid, &sb, 1);
}

inline int V(int semid, unsigned short sem_num) {
  struct sembuf sb = {sem_num, 1, 0};
  return semop(semid, &sb, 1);
}

Worker::Worker(const string &host, const string &port, const string &server_id,
               const string &studio_url, const string &pool_type,
               const string &redis_url, const int max_pool_size,
               const int max_expire_time,
               const unsigned int max_timeout_seconds,
               const unsigned int num_workers)
    : _host(host),
      _port(port),
      _server_id(server_id),
      _main_worker_pid(getpid()),
      _num_workers(std::max(num_workers, 1u)),
      _worker_id(-1),
      _sem_num_per_sem_id(10000),
      _worker_shm_size(1024),
      _max_small_obj_num(calc_max_small_obj_num()),
      _small_obj_size(1000),
      _small_obj_shm_size(1024),
      _worker_shm_name("/worker_" + port),
      _small_obj_shm_name("/small_obj_" + port),
      _func_args_shm_prefix("/args_" + port + "_"),
      _func_result_shm_prefix("/result_" + port + "_"),
      _call_id_counter(0),
      _worker_id_counter(0),
      _enable_logger(calc_enable_logger()),
      _max_timeout_seconds(std::max(max_timeout_seconds, 1u)),
      _enable_py_logger(calc_enable_py_logger()) {
  py::object get_pool =
      py::module::import("agentscope.server.async_result_pool")
          .attr("get_pool");
  _result_pool = get_pool("redis", max_expire_time, max_pool_size, redis_url);

  MAGIC_PREFIX = py::module::import("agentscope.server.servicer")
                     .attr("MAGIC_PREFIX")
                     .cast<string>();

  py::object serialize_lib = py::module::import("agentscope.serialize");
  _serialize = serialize_lib.attr("serialize");
  _deserialize = serialize_lib.attr("deserialize");

  py::object pickle = py::module::import("cloudpickle");
  _pickle_loads = pickle.attr("loads");
  _pickle_dumps = pickle.attr("dumps");

  _rpc_meta = py::module::import("agentscope.rpc.rpc_meta").attr("RpcMeta");
  _py_logger =
      py::module::import("loguru").attr("logger").attr("opt")("depth"_a = -1);
  py::gil_scoped_release release;

  struct stat info;
  if (stat("./logs/", &info) != 0) {
    mkdir("./logs", 0755);
  }

  // init worker shm
  _worker_shm_fd = shm_open(_worker_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (_worker_shm_fd == -1) {
    ERROR("shm_open (" + _worker_shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  ftruncate(_worker_shm_fd, _num_workers * 2 * _worker_shm_size);
  void *worker_shm =
      mmap(NULL, _num_workers * 2 * _worker_shm_size, PROT_READ | PROT_WRITE,
           MAP_SHARED, _worker_shm_fd, 0);
  if (worker_shm == MAP_FAILED) {
    ERROR("mmap (worker_shm)");
    kill(_main_worker_pid, SIGINT);
  }
  _worker_shm = (char *)worker_shm;

  // init small object pool
  _small_obj_shm_fd =
      shm_open(_small_obj_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (_small_obj_shm_fd == -1) {
    ERROR("shm_open (" + _small_obj_shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  ftruncate(_small_obj_shm_fd, _max_small_obj_num * _small_obj_shm_size);
  void *small_obj_shm =
      mmap(NULL, _max_small_obj_num * _small_obj_shm_size,
           PROT_READ | PROT_WRITE, MAP_SHARED, _small_obj_shm_fd, 0);
  if (small_obj_shm == MAP_FAILED) {
    ERROR("mmap (" + _small_obj_shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  _small_obj_shm = (char *)small_obj_shm;
  memset(_small_obj_shm, 0, _max_small_obj_num * _small_obj_shm_size);

  // init worker semaphores
  string filename = "./logs/" + _port + ".log";
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    ERROR("open (" + filename + ")");
    kill(_main_worker_pid, SIGINT);
  }
  close(fd);
  unsigned short *_sem_values = new unsigned short[_sem_num_per_sem_id]();
  int key_id = 0;
  for (int i = 0; i * _sem_num_per_sem_id < _num_workers * 4; i++) {
    key_t key = ftok(filename.c_str(), key_id++);
    if (key == -1) {
      ERROR("ftok (" + filename + ")");
      kill(_main_worker_pid, SIGINT);
    }
    int sem_num = std::min(_sem_num_per_sem_id,
                           _num_workers * 4 - i * _sem_num_per_sem_id);
    int semid = semget(key, sem_num, 0666 | IPC_CREAT);
    if (semid == -1) {
      ERROR("semget (" + to_string(key) + ")");
      kill(_main_worker_pid, SIGINT);
    }
    semctl(semid, 0, SETALL, _sem_values);
    _worker_sem_ids.push_back(semid);
  }

  // init small object semaphores
  for (int i = 0; i < _sem_num_per_sem_id; i++) {
    _sem_values[i] = 1;
  }
  for (int i = 0; i * _sem_num_per_sem_id < _max_small_obj_num; i++) {
    key_t key = ftok(filename.c_str(), key_id++);
    if (key == -1) {
      ERROR("ftok (" + filename + ")");
      kill(_main_worker_pid, SIGINT);
    }
    int sem_num = std::min(_sem_num_per_sem_id,
                           _max_small_obj_num - i * _sem_num_per_sem_id);
    int semid = semget(key, sem_num, 0666 | IPC_CREAT);
    if (semid == -1) {
      ERROR("semget (" + to_string(key) + ")");
      kill(_main_worker_pid, SIGINT);
    }
    semctl(semid, 0, SETALL, _sem_values);
    _small_obj_sem_ids.push_back(semid);
  }
  delete[] _sem_values;

  // launch workers
  for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
    pid_t pid = fork();
    if (pid > 0) {
      // parent process
      _worker_pids.push_back(pid);
      _result_mutexes.push_back(std::make_unique<mutex>());
      _result_cvs.push_back(std::make_unique<condition_variable>());
      _result_maps.push_back(std::make_shared<unordered_map<int, string>>());
      release_set_result(worker_id);
    } else if (pid == 0) {
      // child process
      string filename = "./logs/" + _port + "-" + to_string(worker_id) + ".log";
      int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd == -1) {
        ERROR("open (" + filename + ")");
        kill(_main_worker_pid, SIGINT);
      }
      if (dup2(fd, STDOUT_FILENO) == -1 || dup2(fd, STDERR_FILENO) == -1) {
        ERROR("dup2");
        kill(_main_worker_pid, SIGINT);
      }
      close(fd);

      _worker_id = worker_id;
      char *shm_ptr = _worker_shm + (worker_id * 2) * _worker_shm_size;
      release_set_args();
      while (true) {
        acquire_get_args();
        int call_id = *(int *)shm_ptr;
        function_ids function_id = *(function_ids *)(shm_ptr + sizeof(int));
        int obj_id = *(int *)(shm_ptr + sizeof(int) * 2);
        release_set_args();
        LOG(FORMAT(call_id), FUNC_FORMAT(function_id), FORMAT(obj_id));
        thread work;
        switch (function_id) {
          case function_ids::create_agent: {
            work = thread(&Worker::create_agent_worker, this, call_id, obj_id);
            break;
          }
          case function_ids::delete_agent: {
            work = thread(&Worker::delete_agent_worker, this, call_id, obj_id);
            break;
          }
          case function_ids::delete_all_agents: {
            work = thread(&Worker::delete_all_agents_worker, this, call_id);
            break;
          }
          case function_ids::clone_agent: {
            work = thread(&Worker::clone_agent_worker, this, call_id, obj_id);
            break;
          }
          case function_ids::get_agent_list: {
            work = thread(&Worker::get_agent_list_worker, this, call_id);
            break;
          }
          case function_ids::set_model_configs: {
            work = thread(&Worker::set_model_configs_worker, this, call_id,
                          obj_id);
            break;
          }
          case function_ids::get_agent_memory: {
            work =
                thread(&Worker::get_agent_memory_worker, this, call_id, obj_id);
            break;
          }
          case function_ids::agent_func: {
            work = thread(&Worker::agent_func_worker, this, call_id, obj_id);
            break;
          }
          case function_ids::server_info: {
            work = thread(&Worker::server_info_worker, this, call_id);
            break;
          }
        }
        work.detach();
      }
      kill(_main_worker_pid, SIGINT);
    } else if (pid < 0) {
      ERROR("fork (" + to_string(worker_id) + ")");
      kill(_main_worker_pid, SIGINT);
    }
  }
  auto result_thread = thread(&Worker::wait_result, this);
  result_thread.detach();
}

Worker::~Worker()  // for main process to release resources
{
  // release worker_shm
  close(_worker_shm_fd);
  munmap(_worker_shm, _num_workers * _worker_shm_size);

  // release small object pool
  close(_small_obj_shm_fd);
  munmap(_small_obj_shm, _max_small_obj_num * _small_obj_shm_size);

  if (_main_worker_pid == getpid()) {
    for (auto pid : _worker_pids) {
      kill(pid, SIGINT);
      waitpid(pid, NULL, 0);
    }

    shm_unlink(_worker_shm_name.c_str());
    shm_unlink(_small_obj_shm_name.c_str());
    // release semaphores
    for (auto semid : _worker_sem_ids) {
      semctl(semid, 0, IPC_RMID);
    }
    for (auto semid : _small_obj_sem_ids) {
      semctl(semid, 0, IPC_RMID);
    }

    // release large object shm
    for (auto call_id = 0u; call_id < _max_small_obj_num; call_id++) {
      for (auto prefix : {_func_args_shm_prefix, _func_result_shm_prefix}) {
        string shm_name = prefix + to_string(call_id);
        int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
        if (shm_fd != -1) {
          close(shm_fd);
          shm_unlink(shm_name.c_str());
        }
      }
    }
  } else {
    kill(_main_worker_pid, SIGINT);
  }
}

int Worker::acquire_set_args(const int worker_id, const short sem_flg) {
  return P(_worker_sem_ids[(worker_id * 4) / _sem_num_per_sem_id],
           (worker_id * 4) % _sem_num_per_sem_id, sem_flg);
}

int Worker::release_set_args() {
  return V(_worker_sem_ids[(_worker_id * 4) / _sem_num_per_sem_id],
           (_worker_id * 4) % _sem_num_per_sem_id);
}

int Worker::acquire_get_args() {
  return P(_worker_sem_ids[(_worker_id * 4 + 1) / _sem_num_per_sem_id],
           (_worker_id * 4 + 1) % _sem_num_per_sem_id);
}

int Worker::release_get_args(const int worker_id) {
  return V(_worker_sem_ids[(worker_id * 4 + 1) / _sem_num_per_sem_id],
           (worker_id * 4 + 1) % _sem_num_per_sem_id);
}

int Worker::acquire_set_result() {
  return P(_worker_sem_ids[(_worker_id * 4 + 2) / _sem_num_per_sem_id],
           (_worker_id * 4 + 2) % _sem_num_per_sem_id);
}

int Worker::release_set_result(const int worker_id) {
  return V(_worker_sem_ids[(worker_id * 4 + 2) / _sem_num_per_sem_id],
           (worker_id * 4 + 2) % _sem_num_per_sem_id);
}

int Worker::acquire_get_result(const int worker_id, const short sem_flg) {
  return P(_worker_sem_ids[(worker_id * 4 + 3) / _sem_num_per_sem_id],
           (worker_id * 4 + 3) % _sem_num_per_sem_id, sem_flg);
}

int Worker::release_get_result() {
  return V(_worker_sem_ids[(_worker_id * 4 + 3) / _sem_num_per_sem_id],
           (_worker_id * 4 + 3) % _sem_num_per_sem_id);
}

int Worker::find_avail_worker_id() {
  int worker_id = _worker_id_counter.fetch_add(1) % _num_workers;
  acquire_set_args(worker_id);
  LOG(FORMAT(worker_id));
  return worker_id;
}

int Worker::get_call_id() { return _call_id_counter.fetch_add(1); }

int Worker::get_obj_id(const int call_id, const unsigned int obj_size) {
  if (obj_size > _small_obj_size) {
    return -1;
  }
  for (int obj_id = call_id % _max_small_obj_num;;
       obj_id = (obj_id == _max_small_obj_num - 1 ? 0 : obj_id + 1)) {
    if (P(_small_obj_sem_ids[obj_id / _sem_num_per_sem_id],
          obj_id % _sem_num_per_sem_id, IPC_NOWAIT) == 0) {
      LOG(FORMAT(obj_id));
      return obj_id;
    }
  }
}

string Worker::get_content(const string &prefix, const int call_id,
                           const int obj_id) {
  LOG(FORMAT(prefix), FORMAT(call_id), FORMAT(obj_id));
  if (obj_id >= 0) {
    char *small_obj_shm = _small_obj_shm + obj_id * _small_obj_shm_size;
    LOG(FORMAT(call_id), FORMAT(obj_id));
    int content_size = *(int *)(small_obj_shm);
    string result(small_obj_shm + sizeof(int),
                  small_obj_shm + sizeof(int) + content_size);
    LOG("get_content in pool", FORMAT(content_size), BIN_FORMAT(result));
    V(_small_obj_sem_ids[obj_id / _sem_num_per_sem_id],
      obj_id % _sem_num_per_sem_id);
    return result;
  }
  assert(obj_id == -1);
  string shm_name = prefix + to_string(call_id);
  int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
  if (shm_fd == -1) {
    ERROR("shm_open (" + shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  struct stat shm_stat;
  if (fstat(shm_fd, &shm_stat) == -1) {
    close(shm_fd);
    ERROR("fstat (" + shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  auto shm_size = shm_stat.st_size;
  LOG(FORMAT(shm_name), FORMAT(shm_size));
  void *shm = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (shm == MAP_FAILED) {
    ERROR("mmap (" + shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  int content_size = *(int *)shm;
  string content((char *)shm + sizeof(int),
                 (char *)shm + sizeof(int) + content_size);
  LOG(FORMAT(shm_name), FORMAT(content_size), BIN_FORMAT(content));
  munmap(shm, shm_size);
  close(shm_fd);
  shm_unlink(shm_name.c_str());
  return content;
}

void Worker::set_content(const string &prefix, const int call_id,
                         const int obj_id, const string &content) {
  LOG(FORMAT(obj_id), FORMAT(content.size()), BIN_FORMAT(content));
  if (obj_id >= 0) {
    char *small_obj_shm = _small_obj_shm + obj_id * _small_obj_shm_size;
    *(int *)small_obj_shm = content.size();
    memcpy(small_obj_shm + sizeof(int), content.c_str(), content.size());
    LOG("set_content in pool", FORMAT(obj_id));
    return;
  }
  assert(obj_id == -1);
  string shm_name = prefix + to_string(call_id);
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    ERROR("shm_open (" + shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  LOG(FORMAT(shm_name), BIN_FORMAT(content));
  int shm_size = content.size() + sizeof(int);
  ftruncate(shm_fd, shm_size);
  void *shm =
      mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm == MAP_FAILED) {
    ERROR("mmap (" + shm_name + ")");
    kill(_main_worker_pid, SIGINT);
  }
  *(int *)shm = (int)content.size();
  memcpy((char *)shm + sizeof(int), content.c_str(), content.size());
  munmap(shm, shm_size);
  close(shm_fd);
  LOG("final", FORMAT(shm_name), FORMAT(shm_size));
}

string Worker::get_args_repr(const int call_id, const int obj_id) {
  return get_content(_func_args_shm_prefix, call_id, obj_id);
}

void Worker::set_args_repr(const int call_id, const int obj_id,
                           const string &args_repr) {
  set_content(_func_args_shm_prefix, call_id, obj_id, args_repr);
}

void Worker::wait_result() {
  while (true) {
    for (auto worker_id = 0; worker_id < _num_workers; worker_id++) {
      if (acquire_get_result(worker_id, IPC_NOWAIT) == 0) {
        int call_id =
            *(int *)(_worker_shm + (worker_id * 2 + 1) * _worker_shm_size);
        int obj_id =
            *(int *)(_worker_shm + (worker_id * 2 + 1) * _worker_shm_size +
                     sizeof(int));
        release_set_result(worker_id);
        {
          unique_lock<mutex> lock(*_result_mutexes[worker_id]);
          string result = get_content(_func_result_shm_prefix, call_id, obj_id);
          LOG(FORMAT(call_id), FORMAT(obj_id), FORMAT(result.size()),
              BIN_FORMAT(result));
          _result_maps[worker_id]->insert(make_pair(call_id, result));
        }
        _result_cvs[worker_id]->notify_all();
      }
    }
  }
}

string Worker::get_result(const int call_id, const int worker_id) {
  unique_lock<mutex> lock(*_result_mutexes[worker_id]);
  LOG(FORMAT(call_id));
  auto result_map = _result_maps[worker_id];
  _result_cvs[worker_id]->wait(lock, [result_map, call_id] {
    return result_map->find(call_id) != result_map->end();
  });
  string result = result_map->at(call_id);
  result_map->erase(call_id);
  return result;
}

void Worker::set_result(const int call_id, const string &result) {
  int obj_id = get_obj_id(call_id, result.size());
  set_content(_func_result_shm_prefix, call_id, obj_id, result);
  LOG(FORMAT(call_id), FORMAT(obj_id));
  acquire_set_result();
  *(int *)(_worker_shm + (_worker_id * 2 + 1) * _worker_shm_size) = call_id;
  *(int *)(_worker_shm + (_worker_id * 2 + 1) * _worker_shm_size +
           sizeof(int)) = obj_id;
  release_get_result();
}

int Worker::get_worker_id_by_agent_id(const string &agent_id) {
  shared_lock<shared_mutex> lock(_agent_id_map_mutex);
  if (_agent_id_map.find(agent_id) != _agent_id_map.end()) {
    return _agent_id_map[agent_id];
  } else {
    return -1;
  }
}

int Worker::call_worker_func(const int worker_id, const function_ids func_id,
                             const Message *args, const bool need_wait) {
  if (need_wait) {
    acquire_set_args(worker_id);
  }
  int call_id = get_call_id();
  *(int *)(_worker_shm + (worker_id * 2) * _worker_shm_size) = call_id;
  *(int *)(_worker_shm + (worker_id * 2) * _worker_shm_size + sizeof(int)) =
      func_id;
  *(int *)(_worker_shm + (worker_id * 2) * _worker_shm_size + sizeof(int) * 2) =
      -2;
  LOG(FORMAT(worker_id), FUNC_FORMAT(func_id), FORMAT(call_id));
  if (args != nullptr) {
    string args_str = args->SerializeAsString();
    int obj_id = get_obj_id(call_id, args_str.size());
    set_args_repr(call_id, obj_id, args_str);
    *(int *)(_worker_shm + (worker_id * 2) * _worker_shm_size +
             sizeof(int) * 2) = obj_id;
  }
  release_get_args(worker_id);
  LOG(FORMAT(worker_id), FUNC_FORMAT(func_id), FORMAT(call_id), "finished!");
  return call_id;
}

string Worker::call_create_agent(const string &agent_id,
                                 const string &agent_init_args,
                                 const string &agent_source_code) {
  if (get_worker_id_by_agent_id(agent_id) != -1) {
    string msg = "Agent with agent_id [" + agent_id + "] already exists.";
    LOG(msg);
    py::gil_scoped_acquire acquire;
    PY_LOG("warning", msg)
    return "";
  }
  LOG(FORMAT(agent_id));
  int worker_id = find_avail_worker_id();
  CreateAgentArgs args;
  args.set_agent_id(agent_id);
  args.set_agent_init_args(agent_init_args);
  args.set_agent_source_code(agent_source_code);
  int call_id =
      call_worker_func(worker_id, function_ids::create_agent, &args, false);
  LOG(FORMAT(agent_id), FORMAT(call_id), FORMAT(worker_id));
  string result = get_result(call_id, worker_id);
  if (result.empty()) {
    unique_lock<shared_mutex> lock(_agent_id_map_mutex);
    _agent_id_map.insert(std::make_pair(agent_id, worker_id));
  }
  LOG(FORMAT(agent_id), FORMAT(call_id), FORMAT(worker_id), FORMAT(result));
  return result;
}

void Worker::create_agent_worker(const int call_id, const int obj_id) {
  LOG(FORMAT(call_id));
  string args_repr = get_args_repr(call_id, obj_id);
  CreateAgentArgs args;
  args.ParseFromString(args_repr);
  string agent_id = args.agent_id();
  string agent_init_args = args.agent_init_args();
  string agent_source_code = args.agent_source_code();
  LOG(FORMAT(call_id), FORMAT(agent_id));

  py::gil_scoped_acquire acquire;
  LOG(FORMAT(call_id), FORMAT(agent_id));
  py::object agent_configs = _pickle_loads(py::bytes(agent_init_args));
  string cls_name = agent_configs["class_name"].cast<string>();
  string result;
  py::object agent;
  try {
    py::object cls = _rpc_meta.attr("get_class")(cls_name);
    try {
      agent_configs["kwargs"]["_oid"] = agent_id;
      LOG(FORMAT(call_id), FORMAT(agent_id));
      agent = cls(*agent_configs["args"], **agent_configs["kwargs"]);
    } catch (const std::exception &e) {
      result =
          "Failed to create agent instance <" + cls_name + ">: " + e.what();
      LOG(FORMAT(result));
      PY_LOG("error", result);
    }
  } catch (const std::exception &e) {
    result = "(Class [" + cls_name + "] not found: " + e.what() + ",)";
    LOG(FORMAT(result));
    PY_LOG("error", result);
  }
  if (result.empty()) {
    {
      shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
      unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
      _agent_pool.insert(std::make_pair(agent_id, agent));
    }
    PY_LOG("info",
           "Create agent instance <" + cls_name + ">[" + agent_id + "]");
  }
  LOG(FORMAT(call_id), FORMAT(agent_id), FORMAT(result));
  set_result(call_id, result);
}

string Worker::call_delete_agent(const string &agent_id) {
  int worker_id = get_worker_id_by_agent_id(agent_id);
  if (worker_id == -1) {
    py::gil_scoped_acquire acquire;
    string msg = "Try to delete a non-existent agent [" + agent_id + "].";
    PY_LOG("warning", msg);
    return msg;
  }
  AgentArgs args;
  args.set_agent_id(agent_id);
  int call_id = call_worker_func(worker_id, function_ids::delete_agent, &args);
  {
    unique_lock<shared_mutex> lock(_agent_id_map_mutex);
    _agent_id_map.erase(agent_id);
  }
  string result = get_result(call_id, worker_id);
  return result;
}

void Worker::delete_agent_worker(const int call_id, const int obj_id) {
  string args_repr = get_args_repr(call_id, obj_id);
  AgentArgs args;
  args.ParseFromString(args_repr);
  string agent_id = args.agent_id();

  unique_lock<shared_mutex> lock(_agent_pool_delete_mutex);
  unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
  auto agent = _agent_pool[agent_id];
  py::gil_scoped_acquire acquire;
  string class_name = agent.attr("__class__").attr("__name__").cast<string>();
  if (py::hasattr(agent, "__del__")) {
    agent.attr("__del__")();
  }
  _agent_pool.erase(agent_id);
  PY_LOG("info",
         "delete agent instance <" + class_name + ">[" + agent_id + "]");
  set_result(call_id, "");
}

string Worker::call_delete_all_agents() {
  vector<int> call_id_list;
  {
    unique_lock<shared_mutex> lock(_agent_id_map_mutex);
    for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
      int call_id =
          call_worker_func(worker_id, function_ids::delete_all_agents, nullptr);
      call_id_list.push_back(call_id);
    }
    _agent_id_map.clear();
  }
  string final_result;
  for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
    int call_id = call_id_list[worker_id];
    string result = get_result(call_id, worker_id);
    final_result += result;
  }
  py::gil_scoped_acquire acquire;
  PY_LOG("info", "Deleting all agent instances on the server");
  return final_result;
}

void Worker::delete_all_agents_worker(const int call_id) {
  unique_lock<shared_mutex> lock(_agent_pool_delete_mutex);
  unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
  py::gil_scoped_acquire acquire;
  for (auto &agent : _agent_pool) {
    if (py::hasattr(agent.second, "__del__")) {
      agent.second.attr("__del__")();
    }
  }
  _agent_pool.clear();
  set_result(call_id, "");
}

pair<bool, string> Worker::call_clone_agent(const string &agent_id) {
  int worker_id = get_worker_id_by_agent_id(agent_id);
  if (worker_id == -1) {
    return make_pair(false,
                     "Try to clone a non-existent agent [" + agent_id + "].");
  }
  AgentArgs args;
  args.set_agent_id(agent_id);
  int call_id = call_worker_func(worker_id, function_ids::clone_agent, &args);
  string clone_agent_id = get_result(call_id, worker_id);
  {
    unique_lock<shared_mutex> lock(_agent_id_map_mutex);
    _agent_id_map.insert(std::make_pair(clone_agent_id, worker_id));
  }
  return make_pair(true, clone_agent_id);
}

void Worker::clone_agent_worker(const int call_id, const int obj_id) {
  string args_repr = get_args_repr(call_id, obj_id);
  AgentArgs args;
  args.ParseFromString(args_repr);
  string agent_id = args.agent_id();

  shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
  py::object agent;
  {
    shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    agent = _agent_pool[agent_id];
  }
  py::gil_scoped_acquire acquire;
  py::object agent_class = agent.attr("__class__");
  py::object agent_args = agent.attr("_init_settings")["args"];
  py::object agent_kwargs = agent.attr("_init_settings")["kwargs"];
  py::object clone_agent = agent_class(*agent_args, **agent_kwargs);
  string clone_agent_id = clone_agent.attr("agent_id").cast<string>();
  {
    unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    _agent_pool.insert(std::make_pair(clone_agent_id, clone_agent));
  }
  set_result(call_id, clone_agent_id);
}

string Worker::call_get_agent_list() {
  vector<int> call_id_list;
  {
    shared_lock<shared_mutex> lock(_agent_id_map_mutex);
    for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
      int call_id =
          call_worker_func(worker_id, function_ids::get_agent_list, nullptr);
      call_id_list.push_back(call_id);
      LOG(FORMAT(call_id));
    }
  }
  vector<string> result_list;
  for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
    int call_id = call_id_list[worker_id];
    string result_str = get_result(call_id, worker_id);
    LOG(FORMAT(call_id));
    AgentListReturn result;
    result.ParseFromString(result_str);
    for (const auto &agent_str : result.agent_str_list()) {
      result_list.push_back(agent_str);
    }
  }
  LOG(FORMAT(result_list.size()));
  py::gil_scoped_acquire acquire;
  string final_result = _serialize(result_list).cast<string>();
  return final_result;
}

void Worker::get_agent_list_worker(const int call_id) {
  AgentListReturn result;
  {
    shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
    shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    py::gil_scoped_acquire acquire;
    for (auto &iter : _agent_pool) {
      result.add_agent_str_list(iter.second.attr("__str__")().cast<string>());
    }
  }
  set_result(call_id, result.SerializeAsString());
}

string Worker::call_set_model_configs(const string &model_configs) {
  vector<int> call_id_list;
  ModelConfigsArgs args;
  args.set_model_configs(model_configs);
  for (auto i = 0u; i < _num_workers; i++) {
    int call_id = call_worker_func(i, function_ids::set_model_configs, &args);
    call_id_list.push_back(call_id);
  }
  string final_result;
  for (auto worker_id = 0u; worker_id < _num_workers; worker_id++) {
    int call_id = call_id_list[worker_id];
    string result = get_result(call_id, worker_id);
    final_result += result;
  }
  return final_result;
}

void Worker::set_model_configs_worker(const int call_id, const int obj_id) {
  string args_repr = get_args_repr(call_id, obj_id);
  ModelConfigsArgs args;
  args.ParseFromString(args_repr);
  string model_configs_str = args.model_configs();
  py::gil_scoped_acquire acquire;
  py::object model_configs =
      py::module::import("json").attr("loads")(model_configs_str);
  py::module::import("agentscope.manager")
      .attr("ModelManager")
      .attr("get_instance")()
      .attr("load_model_configs")(model_configs);
  set_result(call_id, "");
}

pair<bool, string> Worker::call_get_agent_memory(const string &agent_id) {
  int worker_id = get_worker_id_by_agent_id(agent_id);
  if (worker_id == -1) {
    return make_pair(
        false, "Try to get memory of a non-existent agent [" + agent_id + "].");
  }
  AgentArgs args;
  args.set_agent_id(agent_id);
  int call_id =
      call_worker_func(worker_id, function_ids::get_agent_memory, &args);
  string result_str = get_result(call_id, worker_id);
  MsgReturn result;
  result.ParseFromString(result_str);
  return make_pair(result.ok(), result.message());
}

void Worker::get_agent_memory_worker(const int call_id, const int obj_id) {
  string args_repr = get_args_repr(call_id, obj_id);
  AgentArgs args;
  args.ParseFromString(args_repr);
  string agent_id = args.agent_id();
  py::object agent;
  shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
  {
    shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    agent = _agent_pool[agent_id];
  }
  py::gil_scoped_acquire acquire;
  py::object memory = agent.attr("memory");
  MsgReturn result;
  if (memory.is_none()) {
    result.set_ok(false);
    result.set_message("Agent [" + agent_id + "] has no memory.");
  } else {
    py::object memory_info = memory.attr("get_memory")();
    string memory_msg = _serialize(memory_info).cast<string>();
    result.set_ok(true);
    result.set_message(memory_msg);
  }
  set_result(call_id, result.SerializeAsString());
}

pair<bool, string> Worker::call_agent_func(const string &agent_id,
                                           const string &func_name,
                                           const string &raw_value) {
  int worker_id = get_worker_id_by_agent_id(agent_id);
  if (worker_id == -1) {
    return make_pair(false, "Agent [" + agent_id + "] not exists..");
  }
  AgentFuncArgs args;
  args.set_agent_id(agent_id);
  args.set_func_name(func_name);
  args.set_raw_value(raw_value);
  LOG(FORMAT(agent_id), FORMAT(func_name), BIN_FORMAT(raw_value));
  int call_id = call_worker_func(worker_id, function_ids::agent_func, &args);
  string result_str = get_result(call_id, worker_id);
  AgentFuncReturn result;
  result.ParseFromString(result_str);
  LOG(FORMAT(agent_id), FORMAT(func_name), FORMAT(result.ok()),
      BIN_FORMAT(result.value()));
  return make_pair(result.ok(), result.value());
}

void Worker::agent_func_worker(const int call_id, const int obj_id) {
  string args_repr = get_args_repr(call_id, obj_id);
  AgentFuncArgs args;
  args.ParseFromString(args_repr);
  string agent_id = args.agent_id();
  string func_name = args.func_name();
  string raw_value = args.raw_value();
  LOG(FORMAT(agent_id), FORMAT(func_name), BIN_FORMAT(raw_value));

  py::object agent;
  shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
  {
    shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    agent = _agent_pool[agent_id];
  }
  LOG(FORMAT(agent_id), FORMAT(func_name), BIN_FORMAT(raw_value));
  AgentFuncReturn return_result;
  py::gil_scoped_acquire acquire;
  try {
    if (agent.attr("__class__").attr("_async_func").contains(func_name)) {
      int task_id = _result_pool.attr("prepare")().cast<int>();
      return_result.set_ok(true);
      return_result.set_value(_pickle_dumps(task_id).cast<string>());
      set_result(call_id, return_result.SerializeAsString());
      // process_task
      py::object args = _pickle_loads(py::bytes(raw_value));
      py::object AsyncResult_class =
          py::module::import("agentscope.rpc").attr("AsyncResult");
      if (py::isinstance(args, AsyncResult_class)) {
        args = args.attr("result")();
      }
      try {
        py::object result;
        if (func_name == "reply") {
          result = agent.attr(func_name.c_str())(args);
        } else {
          result = agent.attr(func_name.c_str())(
              *args.attr("get")("args", py::tuple()),
              **args.attr("get")("kwargs", py::dict()));
        }
        _result_pool.attr("set")(task_id, _pickle_dumps(result));
      } catch (const std::exception &e) {
        string error_msg = "Agent [" + agent_id + "] error: " + e.what();
        _result_pool.attr("set")(task_id, MAGIC_PREFIX + error_msg);
        PY_LOG("error", error_msg);
      }
    } else if (agent.attr("__class__").attr("_sync_func").contains(func_name)) {
      py::object args = _pickle_loads(py::bytes(raw_value));
      py::object result = agent.attr(func_name.c_str())(
          *args.attr("get")("args", py::tuple()),
          **args.attr("get")("kwargs", py::dict()));
      string result_repr = _pickle_dumps(result).cast<string>();
      return_result.set_ok(true);
      return_result.set_value(result_repr);
      set_result(call_id, return_result.SerializeAsString());
    } else {
      py::object result = agent.attr(func_name.c_str());
      return_result.set_ok(true);
      return_result.set_value(_pickle_dumps(result).cast<string>());
      set_result(call_id, return_result.SerializeAsString());
    }
  } catch (const std::exception &e) {
    string error_msg = "Agent [" + agent_id + "] error: " + e.what();
    PY_LOG("error", error_msg);
    return_result.set_ok(false);
    return_result.set_value(error_msg);
    set_result(call_id, return_result.SerializeAsString());
  }
}

pair<bool, string> Worker::call_update_placeholder(const int task_id) {
  LOG(FORMAT(task_id));
  try {
    py::gil_scoped_acquire acquire;
    string result =
        _result_pool.attr("get")(task_id, _max_timeout_seconds).cast<string>();
    if (result.substr(0, MAGIC_PREFIX.size()) == MAGIC_PREFIX) {
      return make_pair(false, result.substr(MAGIC_PREFIX.size()));
    }
    LOG(FORMAT(task_id), BIN_FORMAT(result));
    return make_pair(true, result);
  } catch (const std::exception &e) {
    return make_pair(false, "Timeout");
  }
}

string Worker::call_server_info() {
  int worker_id = find_avail_worker_id();
  int call_id =
      call_worker_func(worker_id, function_ids::server_info, nullptr, false);
  string result = get_result(call_id, worker_id);
  return result;
}

void Worker::server_info_worker(const int call_id) {
  py::gil_scoped_acquire acquire;
  py::object process =
      py::module::import("psutil").attr("Process")(_main_worker_pid);
  double cpu_info =
      process.attr("cpu_percent")("interval"_a = 1).cast<double>();
  double mem_info =
      process.attr("memory_info")().attr("rss").cast<double>() / (1 << 20);
  py::dict result("pid"_a = _main_worker_pid, "id"_a = _server_id,
                  "cpu"_a = cpu_info, "mem"_a = mem_info);
  string result_str =
      py::module::import("json").attr("dumps")(result).cast<string>();
  set_result(call_id, result_str);
}
