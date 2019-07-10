#ifndef UTILS_H
#define UTILS_H

#include "mat.h"
#include "device.h"
#include "allocator.h"
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif
#include <vulkan/vulkan.h>
#include <vector>


namespace backend

	class Allocator;
	class VkAllocator;

	class Option

	{
	public:
		// default option

		Option() {
			lightmode = true;
			num_threads = get_cpu_count();
			blob_allocator = 0;
			workspace_allocator = 0;

			blob_vkallocator = 0;
			workspace_vkallocator = 0;
			staging_vkallocator = 0;
			use_winograd_convolution = true;
			use_sgemm_convolution = true;
			use_int8_inference = true;
			use_vulkan_compute = true;

			use_fp16_packed = true;
			use_fp16_storage = true;
			use_fp16_arithmetic = false;
			use_int8_storage = true;
			use_int8_arithmetic = false;

			// sanitize
			if (num_threads <= 0)
				num_threads = 1;
		}

	public:

		bool lightmode;
		int num_threads;

		Allocator* blob_allocator;
		Allocator* workspace_allocator;
		VkAllocator* blob_vkallocator;
		VkAllocator* workspace_vkallocator;
		VkAllocator* staging_vkallocator;

		bool use_winograd_convolution;
		bool use_sgemm_convolution;
		bool use_int8_inference;
		bool use_vulkan_compute;
		bool use_fp16_packed;
		bool use_fp16_storage;
		bool use_fp16_arithmetic;
		bool use_int8_storage;
		bool use_int8_arithmetic;
	};
}


namespace backend {
#ifdef _WIN32
	class Mutex
	{
	public:
		Mutex() { InitializeSRWLock(&srwlock); }
		~Mutex() {}
		void lock() { AcquireSRWLockExclusive(&srwlock); }
		void unlock() { ReleaseSRWLockExclusive(&srwlock); }
	private:
		friend class ConditionVariable;
		SRWLOCK srwlock;
	};
#else // _WIN32
	class Mutex
	{
	public:
		Mutex() { pthread_mutex_init(&mutex, 0); }
		~Mutex() { pthread_mutex_destroy(&mutex); }
		void lock() { pthread_mutex_lock(&mutex); }
		void unlock() { pthread_mutex_unlock(&mutex); }
	private:
		friend class ConditionVariable;
		pthread_mutex_t mutex;
	};
#endif // _WIN32

	class MutexLockGuard
	{
	public:
		MutexLockGuard(Mutex& _mutex) : mutex(_mutex) { mutex.lock(); }
		~MutexLockGuard() { mutex.unlock(); }
	private:
		Mutex& mutex;
	};

#if _WIN32
	class ConditionVariable
	{
	public:
		ConditionVariable() { InitializeConditionVariable(&condvar); }
		~ConditionVariable() {}
		void wait(Mutex& mutex) { SleepConditionVariableSRW(&condvar, &mutex.srwlock, INFINITE, 0); }
		void broadcast() { WakeAllConditionVariable(&condvar); }
		void signal() { WakeConditionVariable(&condvar); }
	private:
		CONDITION_VARIABLE condvar;
	};
#else // _WIN32
	class ConditionVariable
	{
	public:
		ConditionVariable() { pthread_cond_init(&cond, 0); }
		~ConditionVariable() { pthread_cond_destroy(&cond); }
		void wait(Mutex& mutex) { pthread_cond_wait(&cond, &mutex.mutex); }
		void broadcast() { pthread_cond_broadcast(&cond); }
		void signal() { pthread_cond_signal(&cond); }
	private:
		pthread_cond_t cond;
	};
#endif // _WIN32

#if _WIN32
	static unsigned __stdcall start_wrapper(void* args);
	class Thread
	{
	public:
		Thread(void* (*start)(void*), void* args = 0) { _start = start; _args = args; handle = (HANDLE)_beginthreadex(0, 0, start_wrapper, this, 0, 0); }
		~Thread() {}
		void join() { WaitForSingleObject(handle, INFINITE); CloseHandle(handle); }
	private:
		friend static unsigned __stdcall start_wrapper(void* arg);
		HANDLE handle;
		void* (*_start)(void*);
		void* _args;
	};

	static unsigned __stdcall start_wrapper(void* args)
	{
		Thread* t = (Thread*)args;
		t->_start(t->_args);
		return 0;
	}
#else // _WIN32
	class Thread
	{
	public:
		Thread(void* (*start)(void*), void* args = 0) { pthread_create(&t, 0, start, args); }
		~Thread() {}
		void join() { pthread_join(t, 0); }
	private:
		pthread_t t;
	};
#endif // _WIN32

};

#endif //!UTILS_H