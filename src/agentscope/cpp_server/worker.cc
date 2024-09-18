#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>
#include <random>
#include <errno.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/sem.h>

#include "worker.h"

using std::getenv;
using std::to_string;

using namespace pybind11::literals;

using WorkerArgs::AgentArgs;
using WorkerArgs::CreateAgentArgs;
using WorkerArgs::ModelConfigsArgs;
using WorkerArgs::AgentFuncArgs;
using WorkerArgs::AgentFuncReturn;
using WorkerArgs::AgentListReturn;
using WorkerArgs::MsgReturn;


#define LOG(...) RAW_LOGGER(this, __VA_ARGS__)
#define ERROR_MSG_FORMAT(msg) (string("Error: ") + msg + "failed in " + __FUNCTION__)
#define ERROR(msg) LOG(ERROR_MSG_FORMAT(msg)); perror(ERROR_MSG_FORMAT(msg).c_str())

inline void P(int semid, unsigned short sem_num)
{
    struct sembuf sb = {sem_num, -1, 0};
    semop(semid, &sb, 1);
}

inline void V(int semid, unsigned short sem_num)
{
    struct sembuf sb = {sem_num, 1, 0};
    semop(semid, &sb, 1);
}

Worker::Worker(
    const string &host,
    const string &port,
    const string &server_id,
    const string &studio_url,
    const string &pool_type,
    const string &redis_url,
    const int max_pool_size,
    const int max_expire_time,
    const unsigned int max_timeout_seconds,
    const unsigned int num_workers) : _host(host), _port(port), _server_id(server_id),
                                      _main_worker_pid(getpid()),
                                      _num_workers(std::max(num_workers, 1u)),
                                      _worker_id(-1),
                                      _sem_num_per_sem_id(10000),
                                      _call_shm_size(1024),
                                      _max_call_id(calc_max_call_id()),
                                      _small_obj_size(1000),
                                      _small_obj_shm_size(1024),
                                      _call_worker_shm_name("/call_" + port),
                                      _func_args_shm_prefix("/args_" + port + "_"),
                                      _func_result_shm_prefix("/result_" + port + "_"),
                                      _worker_avail_sem_prefix("/avail_" + port + "_"),
                                      _func_ready_sem_prefix("/func_" + port + "_"),
                                      _small_obj_pool_shm_name("/pool_shm_" + port),
                                      _use_logger(calc_use_logger()),
                                      _max_timeout_seconds(std::max(max_timeout_seconds, 1u))
{
    py::object get_pool = py::module::import("agentscope.server.async_result_pool").attr("get_pool");
    _result_pool = get_pool("redis", max_expire_time, max_pool_size, redis_url);

    MAGIC_PREFIX = py::module::import("agentscope.server.servicer").attr("MAGIC_PREFIX").cast<string>();

    py::object serialize_lib = py::module::import("agentscope.serialize");
    _serialize = serialize_lib.attr("serialize");
    _deserialize = serialize_lib.attr("deserialize");

    py::object pickle = py::module::import("cloudpickle");
    _pickle_loads = pickle.attr("loads");
    _pickle_dumps = pickle.attr("dumps");
    _py_logger = py::module::import("loguru").attr("logger").attr("opt")("depth"_a=-1);
    py::gil_scoped_release release;

    struct stat info;
    if (stat("./logs/", &info) != 0)
    {
        mkdir("./logs", 0755);
    }

    // init call worker shm
    _call_worker_shm_fd = shm_open(_call_worker_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (_call_worker_shm_fd == -1)
    {
        ERROR("shm_open (" + _call_worker_shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    ftruncate(_call_worker_shm_fd, _num_workers * _call_shm_size);
    void *call_worker_shm = mmap(NULL, _num_workers * _call_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, _call_worker_shm_fd, 0);
    if (call_worker_shm == MAP_FAILED)
    {
        ERROR("mmap (call_worker_shm)");
        kill(_main_worker_pid, SIGINT);
    }
    _call_worker_shm = (char *)call_worker_shm;

    // init small object pool
    _small_obj_pool_shm_fd = shm_open(_small_obj_pool_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (_small_obj_pool_shm_fd == -1)
    {
        ERROR("shm_open (" + _small_obj_pool_shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    ftruncate(_small_obj_pool_shm_fd, _max_call_id * _small_obj_shm_size);
    _small_obj_pool_shm = mmap(NULL, _max_call_id * _small_obj_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, _small_obj_pool_shm_fd, 0);
    if (_small_obj_pool_shm == MAP_FAILED)
    {
        ERROR("mmap (" + _small_obj_pool_shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    memset(_small_obj_pool_shm, 0, _max_call_id * _small_obj_shm_size);
    for (auto i = 0u; i < _max_call_id; i++)
    {
        _call_id_pool.push(i);
    }

    // init call semaphores
    string filename = "./logs/" + _port + ".log";
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
    {
        ERROR("open (" + filename + ")");
        kill(_main_worker_pid, SIGINT);
    }
    close(fd);
    unsigned short *_sem_values = new unsigned short[_sem_num_per_sem_id]();
    for (int i = 0; i * _sem_num_per_sem_id < _max_call_id; i++)
    {
        key_t key = ftok(filename.c_str(), i);
        if (key == -1)
        {
            ERROR("ftok (" + filename + ")");
            kill(_main_worker_pid, SIGINT);
        }
        int semid = semget(key, _sem_num_per_sem_id, 0666 | IPC_CREAT);
        if (semid == -1)
        {
            ERROR("semget (" + to_string(key) + ")");
            kill(_main_worker_pid, SIGINT);
        }
        semctl(semid, 0, SETALL, _sem_values);
        _call_sem_ids.push_back(semid);
    }
    delete[] _sem_values;

    // launch workers
    for (auto i = 0u; i < _num_workers; i++)
    {
        string worker_avail_sem_name = _worker_avail_sem_prefix + to_string(i);
        string func_ready_sem_name = _func_ready_sem_prefix + to_string(i);
        sem_t *worker_avail_sem = sem_open(worker_avail_sem_name.c_str(), O_CREAT, 0666, 0);
        sem_t *func_ready_sem = sem_open(func_ready_sem_name.c_str(), O_CREAT, 0666, 0);
        _worker_semaphores.push_back(make_pair(worker_avail_sem, func_ready_sem));

        pid_t pid = fork();
        if (pid > 0)
        {
            // parent process
            _worker_pids.push_back(pid);
        }
        else if (pid == 0)
        {
            // child process
            string filename = "./logs/" + _port + "-" + to_string(i) + ".log";
            int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd == -1)
            {
                ERROR("open (" + filename + ")");
                kill(_main_worker_pid, SIGINT);
            }
            if (dup2(fd, STDOUT_FILENO) == -1 || dup2(fd, STDERR_FILENO) == -1)
            {
                ERROR("dup2");
                kill(_main_worker_pid, SIGINT);
            }
            close(fd);

            _worker_id = i;
            char *shm_ptr = _call_worker_shm + i * _call_shm_size;
            sem_post(worker_avail_sem);
            while (true)
            {
                sem_wait(func_ready_sem);
                int call_id = *(int *)shm_ptr;
                function_ids function_id = *(function_ids *)(shm_ptr + sizeof(int));
                LOG(FORMAT(call_id), FUNC_FORMAT(function_id));
                thread work;
                switch (function_id)
                {
                case function_ids::create_agent:
                {
                    work = thread(&Worker::create_agent_worker, this, call_id);
                    break;
                }
                case function_ids::delete_agent:
                {
                    work = thread(&Worker::delete_agent_worker, this, call_id);
                    break;
                }
                case function_ids::delete_all_agents:
                {
                    work = thread(&Worker::delete_all_agents_worker, this, call_id);
                    break;
                }
                case function_ids::clone_agent:
                {
                    work = thread(&Worker::clone_agent_worker, this, call_id);
                    break;
                }
                case function_ids::get_agent_list:
                {
                    work = thread(&Worker::get_agent_list_worker, this, call_id);
                    break;
                }
                case function_ids::set_model_configs:
                {
                    work = thread(&Worker::set_model_configs_worker, this, call_id);
                    break;
                }
                case function_ids::get_agent_memory:
                {
                    work = thread(&Worker::get_agent_memory_worker, this, call_id);
                    break;
                }
                case function_ids::server_info:
                {
                    work = thread(&Worker::server_info_worker, this, call_id);
                    break;
                }
                case function_ids::agent_func:
                {
                    work = thread(&Worker::agent_func_worker, this, call_id);
                }
                }
                work.detach();
                sem_post(worker_avail_sem);
            }
            kill(_main_worker_pid, SIGINT);
        }
        else if (pid < 0)
        {
            ERROR("fork (" + to_string(i) +")");
            kill(_main_worker_pid, SIGINT);
        }
    }
}

Worker::~Worker() // for main process to release resources
{
    // release call_worker_shm
    close(_call_worker_shm_fd);
    munmap(_call_worker_shm, _num_workers * _call_shm_size);

    // release small object pool
    close(_small_obj_pool_shm_fd);
    munmap(_small_obj_pool_shm, _max_call_id * _small_obj_shm_size);

    // release worker semaphores
    for (auto iter : _worker_semaphores)
    {
        sem_t *worker_avail_sem = iter.first;
        sem_t *func_ready_sem = iter.second;
        sem_close(func_ready_sem);
        sem_close(worker_avail_sem);
    }

    if (_main_worker_pid == getpid())
    {
        for (auto pid : _worker_pids)
        {
            kill(pid, SIGINT);
            waitpid(pid, NULL, 0);
        }

        shm_unlink(_call_worker_shm_name.c_str());
        shm_unlink(_small_obj_pool_shm_name.c_str());
        // release call semaphores
        for (auto semid : _call_sem_ids)
        {
            semctl(semid, 0, IPC_RMID);
        }

        // release worker semaphores
        for (auto i = 0u; i < _worker_semaphores.size(); i++)
        {
            string worker_avail_sem_name = _worker_avail_sem_prefix + to_string(i);
            string func_ready_sem_name = _func_ready_sem_prefix + to_string(i);
            sem_unlink(worker_avail_sem_name.c_str());
            sem_unlink(func_ready_sem_name.c_str());
        }

        // release large object shm
        for (auto call_id = 0u; call_id < _max_call_id; call_id++)
        {
            for (auto prefix : {_func_args_shm_prefix, _func_result_shm_prefix})
            {
                string shm_name = prefix + to_string(call_id);
                int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
                if (shm_fd != -1)
                {
                    close(shm_fd);
                    shm_unlink(shm_name.c_str());
                }
            }
        }
    }
    else
    {
        kill(_main_worker_pid, SIGINT);
    }
}

int Worker::find_avail_worker_id()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, _num_workers - 1);
    int worker_id;
    for (auto cnt = 0u; cnt < 4 * _num_workers; cnt++)
    {
        worker_id = dis(gen);
        if (sem_trywait(_worker_semaphores[worker_id].first) == 0)
        {
            assert(sem_trywait(_worker_semaphores[worker_id].first) != 0);
            LOG(FORMAT(worker_id));
            return worker_id;
        }
    }
    sem_wait(_worker_semaphores[worker_id].first);
    LOG(FORMAT(worker_id));
    return worker_id;
}

int Worker::get_call_id()
{
    unique_lock<mutex> lock(_call_id_mutex);
    _call_id_cv.wait(lock, [this]
                     { return !_call_id_pool.empty(); });
    int call_id = _call_id_pool.front();
    _call_id_pool.pop();
    return call_id;
}

string Worker::get_content(const string &prefix, const int call_id)
{
    char *small_obj_shm = (char *)_small_obj_pool_shm + call_id * _small_obj_shm_size;
    int *occupied = (int *)small_obj_shm;
    LOG(FORMAT(*occupied), FORMAT(call_id));
    if (*occupied)
    {
        int content_size = *(int *)(small_obj_shm + sizeof(int));
        string result(small_obj_shm + sizeof(int) * 2, small_obj_shm + sizeof(int) * 2 + content_size);
        LOG("get_content in pool", FORMAT(content_size), BIN_FORMAT(result));
        *occupied = false;
        return result;
    }
    string shm_name = prefix + to_string(call_id);
    int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1)
    {
        ERROR("shm_open (" + shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    struct stat shm_stat;
    if (fstat(shm_fd, &shm_stat) == -1)
    {
        close(shm_fd);
        ERROR("fstat (" + shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    auto shm_size = shm_stat.st_size;
    LOG(FORMAT(shm_name), FORMAT(shm_size));
    void *shm = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED)
    {
        ERROR("mmap (" + shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    int content_size = *(int *)shm;
    string content((char *)shm + sizeof(int), (char *)shm + sizeof(int) + content_size);
    LOG(FORMAT(shm_name), FORMAT(content_size), BIN_FORMAT(content));
    munmap(shm, shm_size);
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    return content;
}

void Worker::set_content(const string &prefix, const int call_id, const string &content)
{
    LOG(FORMAT(content.size()), BIN_FORMAT(content));
    if (content.size() <= _small_obj_size)
    {
        char *small_obj_shm = (char *)_small_obj_pool_shm + call_id * _small_obj_shm_size;
        *(int *)small_obj_shm = true;
        *(int *)(small_obj_shm + sizeof(int)) = content.size();
        memcpy(small_obj_shm + sizeof(int) * 2, content.c_str(), content.size());
        LOG("set_content in pool", FORMAT(call_id));
        return;
    }
    string shm_name = prefix + to_string(call_id);
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1)
    {
        ERROR("shm_open (" + shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    LOG(FORMAT(shm_name), BIN_FORMAT(content));
    int shm_size = content.size() + sizeof(int);
    ftruncate(shm_fd, shm_size);
    void *shm = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED)
    {
        ERROR("mmap (" + shm_name + ")");
        kill(_main_worker_pid, SIGINT);
    }
    *(int *)shm = (int)content.size();
    memcpy((char *)shm + sizeof(int), content.c_str(), content.size());
    munmap(shm, shm_size);
    close(shm_fd);
    LOG("final", FORMAT(shm_name), FORMAT(shm_size));
}

string Worker::get_args_repr(const int call_id)
{
    return get_content(_func_args_shm_prefix, call_id);
}

void Worker::set_args_repr(const int call_id, const string &args_repr)
{
    set_content(_func_args_shm_prefix, call_id, args_repr);
}

string Worker::get_result(const int call_id)
{
    P(_call_sem_ids[call_id / _sem_num_per_sem_id], call_id % _sem_num_per_sem_id);
    string result = get_content(_func_result_shm_prefix, call_id);
    {
        unique_lock lock(_call_id_mutex);
        _call_id_pool.push(call_id);
        _call_id_cv.notify_one();
    }
    return result;
}

void Worker::set_result(const int call_id, const string &result)
{
    set_content(_func_result_shm_prefix, call_id, result);
    V(_call_sem_ids[call_id / _sem_num_per_sem_id], call_id % _sem_num_per_sem_id);
}

int Worker::get_worker_id_by_agent_id(const string &agent_id)
{
    shared_lock<shared_mutex> lock(_agent_id_map_mutex);
    if (_agent_id_map.find(agent_id) != _agent_id_map.end())
    {
        return _agent_id_map[agent_id];
    }
    else
    {
        return -1;
    }
}

int Worker::call_worker_func(const int worker_id, const function_ids func_id, const Message *args, const bool need_wait)
{
    if (need_wait)
    {
        sem_wait(_worker_semaphores[worker_id].first);
    }
    int call_id = get_call_id();
    *(int *)(_call_worker_shm + worker_id * _call_shm_size) = call_id;
    *(int *)(_call_worker_shm + worker_id * _call_shm_size + sizeof(int)) = func_id;
    LOG(FORMAT(worker_id), FUNC_FORMAT(func_id), FORMAT(call_id));
    if (args != nullptr)
    {
        string args_str = args->SerializeAsString();
        set_args_repr(call_id, args_str);
    }
    sem_post(_worker_semaphores[worker_id].second);
    LOG(FORMAT(worker_id), FUNC_FORMAT(func_id), FORMAT(call_id), "finished!");
    return call_id;
}

string Worker::call_create_agent(const string &agent_id, const string &agent_init_args, const string &agent_source_code)
{
    if (get_worker_id_by_agent_id(agent_id) != -1)
    {
        return "Agent with agent_id [" + agent_id + "] already exists.";
    }
    LOG(FORMAT(agent_id));
    int worker_id = find_avail_worker_id();
    CreateAgentArgs args;
    args.set_agent_id(agent_id);
    args.set_agent_init_args(agent_init_args);
    args.set_agent_source_code(agent_source_code);
    int call_id = call_worker_func(worker_id, function_ids::create_agent, &args, false);
    LOG(FORMAT(agent_id), FORMAT(call_id), FORMAT(worker_id));
    string result = get_result(call_id);
    if (result.empty())
    {
        unique_lock<shared_mutex> lock(_agent_id_map_mutex);
        _agent_id_map.insert(std::make_pair(agent_id, worker_id));
    }
    LOG(FORMAT(agent_id), FORMAT(call_id), FORMAT(worker_id), FORMAT(result));
    return result;
}

void Worker::create_agent_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
    CreateAgentArgs args;
    args.ParseFromString(args_repr);
    string agent_id = args.agent_id();
    string agent_init_args = args.agent_init_args();
    string agent_source_code = args.agent_source_code();
    LOG(FORMAT(call_id), FORMAT(agent_id));

    py::gil_scoped_acquire acquire;
    py::tuple create_result = py::module::import("agentscope.cpp_server").attr("create_agent")(agent_id, py::bytes(agent_init_args), _host, std::atoi(_port.c_str()));
    py::object agent = create_result[0];
    py::object error_msg = create_result[1];
    string result = error_msg.cast<string>();
    if (result.empty())
    {
        shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
        unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
        _agent_pool.insert(std::make_pair(agent_id, agent));
    }
    LOG(FORMAT(call_id), FORMAT(agent_id), FORMAT(result));
    set_result(call_id, result);
}

string Worker::call_delete_agent(const string &agent_id)
{
    int worker_id = get_worker_id_by_agent_id(agent_id);
    if (worker_id == -1)
    {
        py::gil_scoped_acquire acquire;
        string msg = "Try to delete a non-existent agent [" + agent_id + "].";
        _py_logger.attr("warning")(msg);
        return msg;
    }
    AgentArgs args;
    args.set_agent_id(agent_id);
    int call_id = call_worker_func(worker_id, function_ids::delete_agent, &args);
    {
        unique_lock<shared_mutex> lock(_agent_id_map_mutex);
        _agent_id_map.erase(agent_id);
    }
    string result = get_result(call_id);
    return result;
}

void Worker::delete_agent_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
    AgentArgs args;
    args.ParseFromString(args_repr);
    string agent_id = args.agent_id();

    unique_lock<shared_mutex> lock(_agent_pool_delete_mutex);
    unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    auto agent = _agent_pool[agent_id];
    py::gil_scoped_acquire acquire;
    string class_name = agent.attr("__class__").attr("__name__").cast<string>();
    if (py::hasattr(agent, "__del__"))
    {
        agent.attr("__del__")();
    }
    _agent_pool.erase(agent_id);
    _py_logger.attr("info")("delete agent instance <" + class_name + ">[" + agent_id + "]");
    set_result(call_id, "");
}

string Worker::call_delete_all_agents()
{
    vector<int> call_id_list;
    {
        unique_lock<shared_mutex> lock(_agent_id_map_mutex);
        for (auto worker_id = 0u; worker_id < _num_workers; worker_id++)
        {
            int call_id = call_worker_func(worker_id, function_ids::delete_all_agents, nullptr);
            call_id_list.push_back(call_id);
        }
        _agent_id_map.clear();
    }
    string final_result;
    for (auto call_id : call_id_list)
    {
        string result = get_result(call_id);
        final_result += result;
    }
    py::gil_scoped_acquire acquire;
    _py_logger.attr("info")("Deleting all agent instances on the server");
    return final_result;
}

void Worker::delete_all_agents_worker(const int call_id)
{
    unique_lock<shared_mutex> lock(_agent_pool_delete_mutex);
    unique_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
    py::gil_scoped_acquire acquire;
    for (auto &agent : _agent_pool)
    {
        if (py::hasattr(agent.second, "__del__"))
        {
            agent.second.attr("__del__")();
        }
    }
    _agent_pool.clear();
    set_result(call_id, "");
}

pair<bool, string> Worker::call_clone_agent(const string &agent_id)
{
    int worker_id = get_worker_id_by_agent_id(agent_id);
    if (worker_id == -1)
    {
        return make_pair(false, "Try to clone a non-existent agent [" + agent_id + "].");
    }
    AgentArgs args;
    args.set_agent_id(agent_id);
    int call_id = call_worker_func(worker_id, function_ids::clone_agent, &args);
    string clone_agent_id = get_result(call_id);
    {
        unique_lock<shared_mutex> lock(_agent_id_map_mutex);
        _agent_id_map.insert(std::make_pair(clone_agent_id, worker_id));
    }
    return make_pair(true, clone_agent_id);
}

void Worker::clone_agent_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
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

string Worker::call_get_agent_list()
{
    vector<int> call_id_list;
    {
        shared_lock<shared_mutex> lock(_agent_id_map_mutex);
        for (auto worker_id = 0u; worker_id < _num_workers; worker_id++)
        {
            int call_id = call_worker_func(worker_id, function_ids::get_agent_list, nullptr);
            call_id_list.push_back(call_id);
        }
    }
    vector<string> result_list;
    for (auto call_id : call_id_list)
    {
        string result_str = get_result(call_id);
        AgentListReturn result;
        result.ParseFromString(result_str);
        for (const auto &agent_str : result.agent_str_list())
        {
            result_list.push_back(agent_str);
        }
    }
    py::gil_scoped_acquire acquire;
    string final_result = _serialize(result_list).cast<string>();
    return final_result;
}

void Worker::get_agent_list_worker(const int call_id)
{
    AgentListReturn result;
    {
        shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
        shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
        py::gil_scoped_acquire acquire;
        for (auto &iter : _agent_pool)
        {
            result.add_agent_str_list(iter.second.attr("__str__")().cast<string>());
        }
    }
    set_result(call_id, result.SerializeAsString());
}

string Worker::call_set_model_configs(const string &model_configs)
{
    vector<int> call_id_list;
    ModelConfigsArgs args;
    args.set_model_configs(model_configs);
    for (auto i = 0u; i < _num_workers; i++)
    {
        int call_id = call_worker_func(i, function_ids::set_model_configs, &args);
        call_id_list.push_back(call_id);
    }
    string final_result;
    for (auto call_id : call_id_list)
    {
        string result = get_result(call_id);
        final_result += result;
    }
    return final_result;
}

void Worker::set_model_configs_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
    ModelConfigsArgs args;
    args.ParseFromString(args_repr);
    string model_configs_str = args.model_configs();
    py::gil_scoped_acquire acquire;
    py::object model_configs = py::module::import("json").attr("loads")(model_configs_str);
    py::module::import("agentscope.manager").attr("ModelManager").attr("get_instance")().attr("load_model_configs")(model_configs);
    set_result(call_id, "");
}

pair<bool, string> Worker::call_get_agent_memory(const string &agent_id)
{
    int worker_id = get_worker_id_by_agent_id(agent_id);
    if (worker_id == -1)
    {
        return make_pair(false, "Try to get memory of a non-existent agent [" + agent_id + "].");
    }
    AgentArgs args;
    args.set_agent_id(agent_id);
    int call_id = call_worker_func(worker_id, function_ids::get_agent_memory, &args);
    string result_str = get_result(call_id);
    MsgReturn result;
    result.ParseFromString(result_str);
    return make_pair(result.ok(), result.message());
}

void Worker::get_agent_memory_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
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
    if (memory.is_none())
    {
        result.set_ok(false);
        result.set_message("Agent [" + agent_id + "] has no memory.");
    }
    else
    {
        py::object memory_info = memory.attr("get_memory")();
        string memory_msg = _serialize(memory_info).cast<string>();
        result.set_ok(true);
        result.set_message(memory_msg);
    }
    set_result(call_id, result.SerializeAsString());
}

pair<bool, string> Worker::call_agent_func(const string &agent_id, const string &func_name, const string &raw_value)
{
    int worker_id = get_worker_id_by_agent_id(agent_id);
    if (worker_id == -1)
    {
        return make_pair(false, "Agent [" + agent_id + "] not exists..");
    }
    AgentFuncArgs args;
    args.set_agent_id(agent_id);
    args.set_func_name(func_name);
    args.set_raw_value(raw_value);
    LOG(FORMAT(agent_id), FORMAT(func_name), BIN_FORMAT(raw_value));
    int call_id = call_worker_func(worker_id, function_ids::agent_func, &args);
    string result_str = get_result(call_id);
    AgentFuncReturn result;
    result.ParseFromString(result_str);
    LOG(FORMAT(agent_id), FORMAT(func_name), FORMAT(result.ok()), BIN_FORMAT(result.value()));
    return make_pair(result.ok(), result.value());
}

void Worker::agent_func_worker(const int call_id)
{
    string args_repr = get_args_repr(call_id);
    AgentFuncArgs args;
    args.ParseFromString(args_repr);
    string agent_id = args.agent_id();
    string func_name = args.func_name();
    string raw_value = args.raw_value();

    py::object agent;
    shared_lock<shared_mutex> lock(_agent_pool_delete_mutex);
    {
        shared_lock<shared_mutex> insert_lock(_agent_pool_insert_mutex);
        agent = _agent_pool[agent_id];
    }
    LOG(FORMAT(agent_id), FORMAT(func_name), BIN_FORMAT(raw_value));
    AgentFuncReturn return_result;
    py::gil_scoped_acquire acquire;
    try
    {
        if (agent.attr("__class__").attr("_async_func").contains(func_name))
        {
            int task_id = _result_pool.attr("prepare")().cast<int>();
            return_result.set_ok(true);
            return_result.set_value(_pickle_dumps(task_id).cast<string>());
            set_result(call_id, return_result.SerializeAsString());
            // process_task
            py::object args = _pickle_loads(py::bytes(raw_value));
            py::object AsyncResult_class = py::module::import("agentscope.rpc").attr("AsyncResult");
            if (py::isinstance(args, AsyncResult_class))
            {
                args = args.attr("result")();
            }
            try
            {
                py::object result;
                if (func_name == "reply")
                {
                    result = agent.attr(func_name.c_str())(args);
                }
                else
                {
                    result = agent.attr(func_name.c_str())(
                        *args.attr("get")("args", py::tuple()),
                        **args.attr("get")("kwargs", py::dict()));
                }
                _result_pool.attr("set")(task_id, _pickle_dumps(result));
            }
            catch (const std::exception &e)
            {
                string error_msg = "Agent [" + agent_id + "] error: " + e.what();
                _result_pool.attr("set")(task_id, MAGIC_PREFIX + error_msg);
                _py_logger.attr("error")(error_msg);
            }
        }
        else if (agent.attr("__class__").attr("_sync_func").contains(func_name))
        {
            py::object args = _pickle_loads(py::bytes(raw_value));
            py::object result = agent.attr(func_name.c_str())(
                *args.attr("get")("args", py::tuple()),
                **args.attr("get")("kwargs", py::dict()));
            string result_repr = _pickle_dumps(result).cast<string>();
            return_result.set_ok(true);
            return_result.set_value(result_repr);
            set_result(call_id, return_result.SerializeAsString());
        }
        else
        {
            py::object result = agent.attr(func_name.c_str());
            return_result.set_ok(true);
            return_result.set_value(_pickle_dumps(result).cast<string>());
            set_result(call_id, return_result.SerializeAsString());
        }
    }
    catch (const std::exception &e)
    {
        string error_msg = "Agent [" + agent_id + "] error: " + e.what();
        _py_logger.attr("error")(error_msg);
        return_result.set_ok(false);
        return_result.set_value(error_msg);
        set_result(call_id, return_result.SerializeAsString());
    }
}

pair<bool, string> Worker::call_update_placeholder(const int task_id)
{
    LOG(FORMAT(task_id));
    try
    {
        py::gil_scoped_acquire acquire;
        string result = _result_pool.attr("get")(task_id, _max_timeout_seconds).cast<string>();
        if (result.substr(0, MAGIC_PREFIX.size()) == MAGIC_PREFIX)
        {
            return make_pair(false, result.substr(MAGIC_PREFIX.size()));
        }
        LOG(FORMAT(task_id), BIN_FORMAT(result));
        return make_pair(true, result);
    }
    catch (const std::exception &e)
    {
        return make_pair(false, "Timeout");
    }
}

string Worker::call_server_info()
{
    int worker_id = find_avail_worker_id();
    int call_id = call_worker_func(worker_id, function_ids::server_info, nullptr, false);
    string result = get_result(call_id);
    return result;
}

void Worker::server_info_worker(const int call_id)
{
    py::gil_scoped_acquire acquire;
    py::object process = py::module::import("psutil").attr("Process")(_main_worker_pid);
    double cpu_info = process.attr("cpu_percent")("interval"_a = 1).cast<double>();
    double mem_info = process.attr("memory_info")().attr("rss").cast<double>() / (1 << 20);
    py::dict result("pid"_a = _main_worker_pid, "id"_a = _server_id, "cpu"_a = cpu_info, "mem"_a = mem_info);
    string result_str = py::module::import("json").attr("dumps")(result).cast<string>();
    set_result(call_id, result_str);
}