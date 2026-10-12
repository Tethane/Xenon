#include "render/thread_pool.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

static void require(bool condition, const char *message) {
	if (!condition) {
		std::fprintf(stderr, "%s\n", message);
		std::abort();
	}
}

struct Counts {
	int constructed = 0, destroyed = 0, called = 0;
};

// A moved-from object deliberately retains its counter. Destructing it twice
// therefore exposes SBO lifetime bugs even if its owned resource was moved out.
struct SmallMoveOnly {
	Counts *counts;
	std::unique_ptr<int> value;

	explicit SmallMoveOnly(Counts &c) : counts(&c), value(new int(7)) {
		++counts->constructed;
	}

	SmallMoveOnly(SmallMoveOnly &&other) noexcept
		: counts(other.counts), value(std::move(other.value)) {
		++counts->constructed;
	}

	SmallMoveOnly(const SmallMoveOnly &) = delete;

	~SmallMoveOnly() {
		++counts->destroyed;
	}

	void operator()() {
		require(value && *value == 7, "Moved-from callable executed");
		++counts->called;
	}
};

struct LargeMoveOnly : SmallMoveOnly {
	std::array<char, 160> extra{};

	explicit LargeMoveOnly(Counts &c) : SmallMoveOnly(c) {}

	LargeMoveOnly(LargeMoveOnly &&) = default;
};

template <class Callable> static void check_moves() {
	Counts c;
	{
		xn::Task a = xn::Task::make(Callable(c));
		xn::Task b(std::move(a));
		require(!a && b, "Move constructor did not transfer task ownership");
		xn::Task d = xn::Task::make(Callable(c));
		d = std::move(b);
		require(!b && d, "Move assignment did not transfer task ownership");
		d();
		d = std::move(d);
		d();
		xn::Task empty;
		d = std::move(empty);
		require(!d, "Moving an empty task did not clear destination");
	}
	require(c.constructed == c.destroyed, "Task callable destroyed twice or leaked");
	require(c.called == 2, "Task move lost callable");
}

int main() {
	check_moves<SmallMoveOnly>();
	check_moves<LargeMoveOnly>();

	for (unsigned threads : {1u, 2u, 8u}) {
		xn::ThreadPool pool(threads);
		std::atomic<int> done{0};
		// Short-lived groups repeatedly reuse the same stack address. Workers
		// often finish before wait() starts, exercising its old unsafe fast path.
		constexpr int groups = 12000;

		for (int i = 0; i < groups; ++i) {
			xn::TaskGroup group(pool);
			group.run([&] { done.fetch_add(1, std::memory_order_relaxed); });

			if (i % 3 == 0)
				std::this_thread::yield();
			group.wait();
		}
		require(done == groups, "Task-group completion lost work");
		// Nested waits have to help their own pool, especially with one worker.
		xn::TaskGroup outer(pool);
		constexpr int parents = 32, children = 31;

		for (int i = 0; i < parents; ++i)
			outer.run([&] {
				for (int round = 0; round < 24; ++round) {
					xn::TaskGroup inner(pool);

					for (int j = 0; j < children; ++j)
						inner.run([&] { done.fetch_add(1, std::memory_order_relaxed); });
					inner.wait();
				}
			});
		outer.wait();
		require(done == groups + parents * children * 24, "Nested worker wait lost work");
		// Moving a unique capture once per chunk used to empty all but chunk 0.
		std::atomic<int> pixels{0};
		pool.parallel_for(127, 3, [owned = std::make_unique<int>(5), &pixels](int begin, int end) {
			require(owned && *owned == 5, "parallel_for moved its callable repeatedly");
			pixels += end - begin;
		});
		require(pixels == 127, "parallel_for coverage mismatch");
		pixels = 0;
		pool.parallel_for_2d(
			29, 17, 4, [owned = std::make_unique<int>(9), &pixels](int x0, int y0, int x1, int y1) {
				require(owned && *owned == 9, "parallel_for_2d moved its callable repeatedly");
				pixels += (x1 - x0) * (y1 - y0);
			});
		require(pixels == 29 * 17, "parallel_for_2d coverage mismatch");
		{
			xn::TaskGroup scope(pool);
			scope.run([&] { ++done; });
		}
		require(done == groups + parents * children * 24 + 1,
				"Group destruction did not join its tasks");
	}
	// TLS worker IDs belong to a pool. An 8-worker producer must never index
	// an unrelated one-worker consumer's local deque with its own worker ID.
	{
		xn::ThreadPool producer(8), consumer(1);
		std::atomic<int> count{0};
		xn::TaskGroup outer(producer);

		for (int i = 0; i < 128; ++i)
			outer.run([&] {
				xn::TaskGroup inner(consumer);

				for (int j = 0; j < 8; ++j)
					inner.run([&] { ++count; });
				inner.wait();
			});
		outer.wait();
		require(count == 1024, "Cross-pool submission failed");
	}
	// Constructor/destructor stress exposes worker-count publication races.
	for (int repeat = 0; repeat < 80; ++repeat) {
		xn::ThreadPool pool(4);
		xn::TaskGroup group(pool);
		group.run([] {});
		group.wait();
	}

	// A failed callable construction must roll back pending work instead of
	// leaving wait() blocked forever, including during exception unwinding.
	struct ThrowOnCopy {
		ThrowOnCopy() = default;

		ThrowOnCopy(const ThrowOnCopy &) {
			throw std::runtime_error("expected");
		}

		void operator()() {}
	};

	{
		xn::ThreadPool pool(2);
		xn::TaskGroup group(pool);
		ThrowOnCopy task;
		bool caught = false;

		try {
			group.run(task);
		} catch (const std::runtime_error &) {
			caught = true;
		}
		require(caught, "Throwing callable copy was not observed");
		group.wait();
	}
	{
		xn::ThreadPool pool(4);
		std::atomic<int> drained{0};
		xn::TaskGroup group(pool);
		group.run([] { throw std::runtime_error("task failure"); });

		for (int i = 0; i < 400; ++i)
			group.run([&] { ++drained; });
		bool caught = false;

		try {
			group.wait();
		} catch (const std::runtime_error &error) {
			caught = std::string(error.what()) == "task failure";
		}
		require(caught, "Task execution failure was swallowed");
		require(drained == 400, "wait rethrew before draining pending tasks");
		// Destruction of the failed group must not terminate this process.
		caught = false;
		drained = 0;

		try {
			pool.parallel_for(80, 1, [&](int begin, int) {
				if (begin == 0)
					throw std::runtime_error("parallel failure");
				++drained;
			});
		} catch (const std::runtime_error &) {
			caught = true;
		}
		require(caught && drained == 79, "parallel_for did not join before propagating failure");
		// Worker-helping wait propagates failures too, through the outer group.
		xn::TaskGroup outer(pool);
		outer.run([&] {
			xn::TaskGroup inner(pool);
			inner.run([] { throw std::runtime_error("nested failure"); });
			inner.wait();
		});
		caught = false;

		try {
			outer.wait();
		} catch (const std::runtime_error &) {
			caught = true;
		}
		require(caught, "Nested worker wait swallowed task failure");
	}
	std::puts("Task moves, group lifetimes, nested waits, queue publication and cross-pool stress "
			  "passed");
}
