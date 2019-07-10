#ifndef UTILS_H
#define UTILS_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif
#ifdef _WIN32


#define MAX_PARAM_COUNT 20
#include <vulkan/vulkan.h>
#include <vector>

#include "mat.h"

namespace backend {
	class Mutex
	{
	public:
		Mutex() { InitializeSRWLock(&srwlock); }
		~Mutex() {}
		void lock() { AcquireSRWLockExclusive(&srwlock); }
		void unlock() { ReleaseSRWLockExclusive(&srwlock); }
	private:
		friend class ConditionVariable;
		// NOTE SRWLock is available from windows vista
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

	int create_gpu_instance();
	void destroy_gpu_instance();

	extern int support_VK_KHR_get_physical_device_properties2;
	extern int support_VK_EXT_debug_utils;

	extern PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR;
	extern PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR;
	extern PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR;
	extern PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR;
	extern PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR;
	extern PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
	extern PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

	int get_gpu_count();
	int get_default_gpu_index();

	class GpuInfo
	{
	public:
		VkPhysicalDevice physical_device;

		uint32_t api_version;
		uint32_t driver_version;
		uint32_t vendor_id;
		uint32_t device_id;
		uint8_t pipeline_cache_uuid[VK_UUID_SIZE];

		int type;

		uint32_t max_shared_memory_size;
		uint32_t max_workgroup_count[3];
		uint32_t max_workgroup_invocations;
		uint32_t max_workgroup_size[3];
		size_t memory_map_alignment;
		size_t buffer_offset_alignment;
		float timestamp_period;

		uint32_t compute_queue_family_index;
		uint32_t transfer_queue_family_index;

		uint32_t compute_queue_count;
		uint32_t transfer_queue_count;

		uint32_t unified_memory_index;
		uint32_t device_local_memory_index;
		uint32_t host_visible_memory_index;

		bool support_fp16_packed;
		bool support_fp16_storage;
		bool support_fp16_arithmetic;
		bool support_int8_storage;
		bool support_int8_arithmetic;

		int support_VK_KHR_8bit_storage;
		int support_VK_KHR_16bit_storage;
		int support_VK_KHR_bind_memory2;
		int support_VK_KHR_dedicated_allocation;
		int support_VK_KHR_descriptor_update_template;
		int support_VK_KHR_get_memory_requirements2;
		int support_VK_KHR_push_descriptor;
		int support_VK_KHR_shader_float16_int8;
		int support_VK_KHR_shader_float_controls;
		int support_VK_KHR_storage_buffer_storage_class;
	};

	const GpuInfo& get_gpu_info(int device_index = get_default_gpu_index());

	class VulkanDevice
	{
	public:
		VulkanDevice(int device_index = get_default_gpu_index());
		~VulkanDevice();
		const GpuInfo& info;
		VkDevice vkdevice() const { return device; }
		VkShaderModule get_shader_module(const char* name) const;
		VkShaderModule compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const;
		VkQueue acquire_queue(uint32_t queue_family_index) const;
		void reclaim_queue(uint32_t queue_family_index, VkQueue queue) const;
		VkAllocator* acquire_blob_allocator() const;
		void reclaim_blob_allocator(VkAllocator* allocator) const;
		VkAllocator* acquire_staging_allocator() const;
		void reclaim_staging_allocator(VkAllocator* allocator) const;
		PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
		PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
		PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;
		PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
		PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
		PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR;
		PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;
		PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

	protected:
		int create_shader_module();
		void destroy_shader_module();
		int init_device_extension();

	private:
		VkDevice device;
		std::vector<VkShaderModule> shader_modules;
		mutable std::vector<VkQueue> compute_queues;
		mutable std::vector<VkQueue> transfer_queues;
		mutable Mutex queue_lock;
		mutable std::vector<VkAllocator*> blob_allocators;
		mutable Mutex blob_allocator_lock;
		mutable std::vector<VkAllocator*> staging_allocators;
		mutable Mutex staging_allocator_lock;
	};

	VulkanDevice* get_gpu_device(int device_index = get_default_gpu_index());
	
	class ParamDict
	{
	public:
		ParamDict();
		int get(int id, int def) const;
		float get(int id, float def) const;
		Mat get(int id, const Mat& def) const;

		void set(int id, int i);
		void set(int id, float f);
		void set(int id, const Mat& v);

	protected:
		friend class Net;

		void clear();

		int load_param(const unsigned char*& mem);

	protected:
		struct
		{
			int loaded;
			union { int i; float f; };
			Mat v;
		} params[MAX_PARAM_COUNT];
	};

	class Option
	{
	public:
		Option();

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


};
#endif //!UTILS_H