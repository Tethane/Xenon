#pragma once
// render/thread_pool.h — Simple thread pool for parallel execution

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace xn {

class ThreadPool {
public:
    ThreadPool(uint32_t num_threads) : stop_(false), active_tasks_(0) {
        for (uint32_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                    active_tasks_--;
                    if (active_tasks_ == 0) wait_condition_.notify_all();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
            active_tasks_++;
        }
        condition_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        wait_condition_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
    }

    uint32_t size() const { return (uint32_t)workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;
    std::atomic<bool> stop_;
    std::atomic<int> active_tasks_;
};

} // namespace xn
