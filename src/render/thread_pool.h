#pragma once
// render/thread_pool.h — Simple thread pool for parallel execution
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

namespace xn {

// -------------------------
// Small, move-only task (SBO)
// -------------------------
class Task {
  public:
	Task() noexcept : call_(nullptr), destroy_(nullptr), move_(nullptr) {}

	Task(Task &&other) noexcept {
		other.move_to(*this);
		other.reset();
	}

	Task &operator=(Task &&other) noexcept {
		if (this != &other) {
			reset();
			other.move_to(*this);
			other.reset();
		}

		return *this;
	}

	Task(const Task &) = delete;
	Task &operator=(const Task &) = delete;

	~Task() {
		reset();
	}

	explicit operator bool() const noexcept {
		return call_ != nullptr;
	}

	void operator()() noexcept {
		call_(storage_ptr());
	}

	template <class F> static Task make(F &&f) {
		using Fn = std::decay_t<F>;
		Task t;

		if constexpr (sizeof(Fn) <= kStorage && alignof(Fn) <= alignof(Storage) &&
					  std::is_nothrow_move_constructible_v<Fn>) {
			// In-place
			new (t.storage_ptr()) Fn(std::forward<F>(f));
			t.call_ = [](void *p) noexcept { (*reinterpret_cast<Fn *>(p))(); };
			t.destroy_ = [](void *p) noexcept { reinterpret_cast<Fn *>(p)->~Fn(); };
			t.move_ = [](void *src, void *dst) noexcept {
				new (dst) Fn(std::move(*reinterpret_cast<Fn *>(src)));
				// Task's move operation resets the source exactly once afterwards.
			};

			return t;
		} else {
			// Heap fallback (rare if lambdas are small)
			Fn *heap = new Fn(std::forward<F>(f));
			new (t.storage_ptr()) Fn *(heap);
			t.call_ = [](void *p) noexcept { (**reinterpret_cast<Fn **>(p))(); };
			t.destroy_ = [](void *p) noexcept { delete *reinterpret_cast<Fn **>(p); };
			t.move_ = [](void *src, void *dst) noexcept {
				new (dst) Fn *(std::exchange(*reinterpret_cast<Fn **>(src), nullptr));
			};

			return t;
		}
	}

  private:
	static constexpr size_t kStorage = 64; // tweak: 64 or 128
	using Storage = std::aligned_storage_t<kStorage, alignof(std::max_align_t)>;

	void *storage_ptr() noexcept {
		return &storage_;
	}

	const void *storage_ptr() const noexcept {
		return &storage_;
	}

	void reset() noexcept {
		if (destroy_)
			destroy_(storage_ptr());
		call_ = nullptr;
		destroy_ = nullptr;
		move_ = nullptr;
	}

	void move_to(Task &dst) noexcept {
		dst.call_ = call_;
		dst.destroy_ = destroy_;
		dst.move_ = move_;

		if (move_)
			move_(storage_ptr(), dst.storage_ptr());
	}

	Storage storage_;
	void (*call_)(void *) noexcept;
	void (*destroy_)(void *) noexcept;
	void (*move_)(void *, void *) noexcept;
};

// Forward decl
class ThreadPool;

// -------------------------
// TaskGroup: run tasks + wait (with worker helping)
// -------------------------
class TaskGroup {
  public:
	explicit TaskGroup(ThreadPool &pool) : pool_(&pool), remaining_(0) {}

	// Explicit wait() reports task failures after joining every task. Destruction
	// still joins during unwinding without throwing a second exception.
	~TaskGroup() {
		try {
			wait();
		} catch (...) {
		}
	}

	TaskGroup(const TaskGroup &) = delete;
	TaskGroup &operator=(const TaskGroup &) = delete;

	template <class F> void run(F &&f);

	void wait();

  private:
	ThreadPool *pool_;
	std::atomic<int> remaining_;
	std::mutex m_;
	std::condition_variable cv_;
	std::exception_ptr exception_;

	void task_done() {
		// Publishing zero and waking waiters form one lifetime boundary. A waiter
		// must acquire this mutex before returning, including its zero-work path.
		std::lock_guard<std::mutex> lk(m_);

		if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			cv_.notify_all();
		}
	}

	friend class ThreadPool;
};

// -------------------------
// ThreadPool: work-stealing deque per worker
// -------------------------
class ThreadPool {
  public:
	explicit ThreadPool(uint32_t num_threads = std::thread::hardware_concurrency())
		: stop_(false), next_queue_(0), approx_tasks_(0) {
		if (num_threads == 0)
			num_threads = 1;
		queues_.reserve(num_threads);

		for (uint32_t i = 0; i < num_threads; ++i)
			queues_.push_back(std::make_unique<Queue>());
		workers_.reserve(num_threads);

		for (uint32_t i = 0; i < num_threads; ++i) {
			workers_.emplace_back([this, i] { worker_loop(i); });
		}
	}

	~ThreadPool() {
		stop_.store(true, std::memory_order_release);
		{
			std::lock_guard<std::mutex> lk(cv_m_);
			cv_.notify_all();
		}

		for (auto &t : workers_)
			t.join();
	}

	ThreadPool(const ThreadPool &) = delete;
	ThreadPool &operator=(const ThreadPool &) = delete;

	// Queues are completely constructed before any worker starts. workers_.size()
	// would race with constructor emplace_back calls from early worker threads.
	uint32_t size() const noexcept {
		return (uint32_t)queues_.size();
	}

	// Returns false if pool is stopping/stopped.
	template <class F> bool enqueue(F &&f) {
		if (stop_.load(std::memory_order_acquire))
			return false;

		Task task = Task::make(std::forward<F>(f));

		// Prefer local queue if called from worker
		int wid = tls_pool_ == this ? tls_worker_id_ : -1;

		if (wid >= 0) {
			push_local((uint32_t)wid, std::move(task));
		} else {
			// Round-robin injection
			uint32_t q = next_queue_.fetch_add(1, std::memory_order_relaxed) % size();
			push_local(q, std::move(task));
		}

		// Wake one sleeping worker
		{
			std::lock_guard<std::mutex> lk(cv_m_);
			cv_.notify_one();
		}

		return true;
	}

	// Splits [0,n) into grains and waits for completion. Tasks share fn; its call
	// operator and captures must support concurrent invocation.
	template <class F> void parallel_for(int n, int grain, F &&fn) {
		if (n <= 0)
			return;

		if (grain <= 0)
			grain = 256;

		TaskGroup g(*this);

		for (int b = 0; b < n; b += grain) {
			int e = std::min(b + grain, n);
			// The call waits for every task before fn leaves scope. Share this callable
			// instead of repeatedly moving from an rvalue (which empties its captures).
			g.run([b, e, &fn] { fn(b, e); });
		}
		g.wait();
	}

	// Tiled 2D work shares the callable with the same concurrency requirement.
	template <class F> void parallel_for_2d(int width, int height, int tile, F &&fn) {
		if (width <= 0 || height <= 0)
			return;

		if (tile <= 0)
			tile = 16;

		TaskGroup g(*this);

		for (int ty = 0; ty < height; ty += tile) {
			for (int tx = 0; tx < width; tx += tile) {
				int x0 = tx, y0 = ty;
				int x1 = std::min(tx + tile, width);
				int y1 = std::min(ty + tile, height);
				g.run([=, &fn] { fn(x0, y0, x1, y1); });
			}
		}
		g.wait();
	}

  private:
	struct Queue {
		std::mutex m;
		std::deque<Task> dq;
	};

	std::vector<std::thread> workers_;
	std::vector<std::unique_ptr<Queue>> queues_;

	std::atomic<bool> stop_;
	std::atomic<uint32_t> next_queue_;
	std::atomic<int> approx_tasks_; // approximate number of queued tasks

	std::mutex cv_m_;
	std::condition_variable cv_;

	static thread_local int tls_worker_id_;
	static thread_local ThreadPool *tls_pool_;

	void push_local(uint32_t q, Task &&t) {
		Queue &Q = *queues_[q];
		std::lock_guard<std::mutex> lk(Q.m);
		Q.dq.emplace_back(std::move(t));
		// Count work before unlocking makes it visible to a pop/steal. Publishing
		// afterwards allows a worker to decrement a task not yet counted.
		approx_tasks_.fetch_add(1, std::memory_order_release);
	}

	bool pop_local(uint32_t q, Task &out) {
		Queue &Q = *queues_[q];
		std::lock_guard<std::mutex> lk(Q.m);

		if (Q.dq.empty())
			return false;
		out = std::move(Q.dq.back());
		Q.dq.pop_back();

		return true;
	}

	bool steal(uint32_t thief, Task &out) {
		// Steal from the front of other deques
		const uint32_t n = size();
		// Simple linear probe; for ultra-hot cases you can randomize start.
		for (uint32_t k = 1; k < n; ++k) {
			uint32_t victim = (thief + k) % n;
			Queue &Q = *queues_[victim];
			std::lock_guard<std::mutex> lk(Q.m);

			if (!Q.dq.empty()) {
				out = std::move(Q.dq.front());
				Q.dq.pop_front();

				return true;
			}
		}

		return false;
	}

	bool try_get_task(uint32_t wid, Task &out) {
		if (pop_local(wid, out))
			return true;

		return steal(wid, out);
	}

	void worker_loop(uint32_t wid) {
		tls_pool_ = this;
		tls_worker_id_ = (int)wid;

		while (true) {
			if (stop_.load(std::memory_order_acquire)) {
				// Drain remaining work before exit (optional but usually desirable)
				Task t;

				if (try_get_task(wid, t)) {
					approx_tasks_.fetch_sub(1, std::memory_order_acq_rel);
					t();
					continue;
				}
				break;
			}

			Task task;

			if (try_get_task(wid, task)) {
				approx_tasks_.fetch_sub(1, std::memory_order_acq_rel);
				task();
				continue;
			}

			// Sleep until there's probably work, or stop
			std::unique_lock<std::mutex> lk(cv_m_);
			cv_.wait(lk, [this] {
				return stop_.load(std::memory_order_acquire) ||
					   approx_tasks_.load(std::memory_order_acquire) > 0;
			});
		}

		tls_worker_id_ = -1;
		tls_pool_ = nullptr;
	}

	// Let TaskGroup "help" run tasks while waiting
	void help_one() {
		int wid = tls_worker_id_;

		if (tls_pool_ != this || wid < 0)
			return;

		Task t;

		if (try_get_task((uint32_t)wid, t)) {
			approx_tasks_.fetch_sub(1, std::memory_order_acq_rel);
			t();
		} else {
			// nothing to do; yield a little
			std::this_thread::yield();
		}
	}

	friend class TaskGroup;
};

inline thread_local int ThreadPool::tls_worker_id_ = -1;
inline thread_local ThreadPool *ThreadPool::tls_pool_ = nullptr;

// -------------------------
// TaskGroup implementation
// -------------------------
template <class F> void TaskGroup::run(F &&f) {
	remaining_.fetch_add(1, std::memory_order_release);

	bool ok;

	try {
		ok = pool_->enqueue([this, fn = std::forward<F>(f)]() mutable {
			try {
				fn();
			} catch (...) {
				std::lock_guard<std::mutex> lk(m_);

				if (!exception_)
					exception_ = std::current_exception();
			}
			task_done();
		});
	} catch (...) {
		// Allocation or callable construction failed before queue publication.
		task_done();
		throw;
	}

	if (!ok) {
		// Pool stopped: mark done immediately
		task_done();
	}
}

inline void TaskGroup::wait() {
	// If called from a worker thread, help execute tasks until done
	if (ThreadPool::tls_pool_ == pool_) {
		for (;;) {
			{
				std::lock_guard<std::mutex> lk(m_);

				if (remaining_.load(std::memory_order_acquire) == 0) {
					if (exception_)
						std::rethrow_exception(exception_);

					return;
				}
			}
			pool_->help_one();
		}
	}

	// External thread: block
	std::unique_lock<std::mutex> lk(m_);
	cv_.wait(lk, [this] { return remaining_.load(std::memory_order_acquire) == 0; });

	if (exception_)
		std::rethrow_exception(exception_);
}

} // namespace xn
