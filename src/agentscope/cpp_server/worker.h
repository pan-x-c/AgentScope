#ifndef WORKER_H
#define WORKER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <deque>
#include <queue>
#include <condition_variable>
#include <semaphore.h>
#include <sys/ipc.h>
#include <ctime>
#include <chrono>
#include <atomic>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

#include "worker_args.pb.h"

using std::deque;
using std::queue;
using std::make_pair;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;
using std::atomic;

using std::condition_variable;
using std::condition_variable_any;
using std::mutex;
using std::shared_lock;
using std::shared_mutex;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using std::shared_ptr;

namespace py = pybind11;

using google::protobuf::Message;


#ifdef DEBUG
#define RAW_LOGGER(worker, ...) worker->logger(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define RAW_LOGGER(worker, ...)
#endif

#define FORMAT(var) #var, "=", var
#define BIN_FORMAT(var) #var, "=", bin_format(var)
#define FUNC_FORMAT(var) "worker_func =", Worker::function_ids_to_str(var)

inline string bin_format(const string &var)
{
    bool is_output_to_terminal = isatty(fileno(stdout));
    string converted = var;
    for (char &ch : converted)
    {
        if (!std::isprint(static_cast<unsigned char>(ch)))
        {
            ch = ' ';
        }
    }
    return is_output_to_terminal ? "\033[1;31m" + converted + "\033[1;37m" : "[" + converted + "]";
}


class Worker
{
private:
    const string _host;
    const string _port;
    const string _server_id;
    const int _main_worker_pid;
    const unsigned int _num_workers;
    int _worker_id;
    vector<pid_t> _worker_pids;

    const unsigned int _sem_num_per_sem_id;
    const unsigned int _worker_shm_size;
    const unsigned int _max_small_obj_num;
    const unsigned int _small_obj_size;
    const unsigned int _small_obj_shm_size;

    const string _worker_shm_name;
    const string _small_obj_shm_name;
    const string _func_args_shm_prefix;
    const string _func_result_shm_prefix;

    atomic<int> _call_id_counter;
    atomic<int> _worker_id_counter;

    vector<int> _worker_sem_ids;
    int _worker_shm_fd;
    char *_worker_shm;

    vector<int> _small_obj_sem_ids;
    int _small_obj_shm_fd;
    char *_small_obj_shm;

    vector<unique_ptr<mutex>> _result_mutexes;
    vector<unique_ptr<condition_variable>> _result_cvs;
    vector<shared_ptr<unordered_map<int, string>>> _result_maps;

    const bool _use_logger;
    mutex _logger_mutex;

    unordered_map<string, int> _agent_id_map; // map agent id to worker id
    shared_mutex _agent_id_map_mutex;
    unordered_map<string, py::object> _agent_pool;
    shared_mutex _agent_pool_insert_mutex;
    shared_mutex _agent_pool_delete_mutex;

    const unsigned int _max_timeout_seconds;
    py::object _result_pool;

    // common used functions
    string MAGIC_PREFIX;
    py::object _serialize, _deserialize;
    py::object _pickle_loads, _pickle_dumps;
    py::object _py_logger;

    enum function_ids
    {
        create_agent = 0,
        delete_agent,
        delete_all_agents,
        clone_agent,
        get_agent_list,
        set_model_configs,
        get_agent_memory,
        agent_func,
        server_info,
    };
    inline static string function_ids_to_str(const function_ids func_id)
    {
        switch (func_id)
        {
        case create_agent:
            return "create_agent";
        case delete_agent:
            return "delete_agent";
        case delete_all_agents:
            return "delete_all_agents";
        case clone_agent:
            return "clone_agent";
        case get_agent_list:
            return "get_agent_list";
        case set_model_configs:
            return "set_model_configs";
        case get_agent_memory:
            return "get_agent_memory";
        case agent_func:
            return "agent_func";
        case server_info:
            return "server_info";
        default:
            return "unknown";
        }
    }

    int acquire_set_args(const int worker_id, const short sem_flg = 0);
    int release_set_args();
    int acquire_get_args();
    int release_get_args(const int worker_id);
    int acquire_set_result();
    int release_set_result(const int worker_id);
    int acquire_get_result(const int worker_id, const short sem_flg = 0);
    int release_get_result();
    int find_avail_worker_id();
    int get_call_id();
    int get_obj_id(const int call_id, const unsigned int obj_size);
    string get_content(const string &prefix, const int call_id, const int obj_id);
    void set_content(const string &prefix, const int call_id, const int obj_id, const string &content);
    string get_args_repr(const int call_id, const int obj_id);
    void set_args_repr(const int call_id, const int obj_id, const string &args_repr);
    void wait_result();
    string get_result(const int call_id, const int worker_id);
    void set_result(const int call_id, const string &result);
    int get_worker_id_by_agent_id(const string &agent_id);

    static const unsigned int _default_max_small_obj_num = 10000;
    static unsigned int calc_max_small_obj_num()
    {
        char *max_small_obj_num = getenv("AGENTSCOPE_MAX_SMALL_OBJ_NUM");
        if (max_small_obj_num != nullptr)
        {
            try {
                return std::stoi(max_small_obj_num);
            } catch (const std::invalid_argument&) {
                return _default_max_small_obj_num;
            } catch (const std::out_of_range&) {
                return _default_max_small_obj_num;
            }
        }
        else
        {
            return _default_max_small_obj_num;
        }
    }
    static bool calc_use_logger()
    {
        char *use_logger = getenv("AGENTSCOPE_USE_CPP_LOGGER");
        return (use_logger != nullptr && std::string(use_logger) == "True");
    }

    inline long long get_current_timestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    }
    inline bool is_timeout(const long long timestamp)
    {
        return get_current_timestamp() - timestamp > _max_timeout_seconds;
    }
    int call_worker_func(const int worker_id, const function_ids func_id, const Message *args, const bool need_wait = true);

    void create_agent_worker(const int call_id, const int obj_id);
    void delete_agent_worker(const int call_id, const int obj_id);
    void delete_all_agents_worker(const int call_id);
    void clone_agent_worker(const int call_id, const int obj_id);
    void get_agent_list_worker(const int call_id);
    void set_model_configs_worker(const int call_id, const int obj_id);
    void get_agent_memory_worker(const int call_id, const int obj_id);
    void agent_func_worker(const int call_id, const int obj_id);
    void server_info_worker(const int call_id);

public:
    Worker(
        const string &host,
        const string &port,
        const string &server_id,
        const string &studio_url,
        const string &pool_type,
        const string &redis_url,
        const int max_pool_size,
        const int max_expire_time,
        const unsigned int max_timeout_seconds,
        const unsigned int num_workers);
    ~Worker();

    template<typename... Args>
    void logger(const string &file_name, const string &func_name, const int line_num, Args... args)
    {
        if (_use_logger)
        {
            unique_lock<std::mutex> lock(_logger_mutex);
            bool is_output_to_terminal = isatty(fileno(stdout));

            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            std::tm* localTime = std::localtime(&now_c);
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::cout << (is_output_to_terminal ? "\033[0;32m" : "");
            std::cout << std::put_time(localTime, "%Y-%m-%d %H:%M:%S") 
                << '.' << std::setw(3) << std::setfill('0') << milliseconds.count();
            std::cout << (is_output_to_terminal ? "\033[0m" : "");
            std::cout << " | ";

            std::cout << "port = " << _port << " worker_id = " << std::setw(3) << std::setfill(' ') << _worker_id
                << " tid = " << std::this_thread::get_id() << " | ";

            std::cout << (is_output_to_terminal ? "\033[0;36m" : "");
            std::cout << file_name << ":" << line_num << " " << func_name;
            std::cout << (is_output_to_terminal ? "\033[0m" : "");

            std::string delimiter = " ";
            std::stringstream result;
            ((result << args << delimiter), ...);
            std::string msg = result.str();
            if (!msg.empty()) {
                msg.erase(msg.size() - delimiter.size());
            }
            std::cout << " - " << (is_output_to_terminal ? "\033[1;37m" : "") << msg << (is_output_to_terminal ? "\033[0m" : "") << std::endl;
        }
    }

    string call_create_agent(const string &agent_id, const string &agent_init_args, const string &agent_source_code);
    string call_delete_agent(const string &agent_id);
    string call_delete_all_agents();
    pair<bool, string> call_clone_agent(const string &agent_id);
    string call_get_agent_list();
    string call_set_model_configs(const string &model_configs);
    pair<bool, string> call_get_agent_memory(const string &agent_id);
    pair<bool, string> call_agent_func(const string &agent_id, const string &func_name, const string &raw_value);
    pair<bool, string> call_update_placeholder(const int task_id);
    string call_server_info();
};

#endif
