#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <Windows.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <vulkan/vulkan.h>

#include "utils.hpp"
#endif // !ALLOCATOR_H
#include <intrin.h>

#define MALLOC_ALIGN 16
#define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)


namespace backend {

	static inline size_t alignSize(size_t sz, int n) { return (sz + n - 1) & -n; }
	template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp)){ return (_Tp*)(((size_t)ptr + n - 1) & -n); }

	static inline void* fastMalloc(size_t size) {
		return _aligned_malloc(size, MALLOC_ALIGN);
	}
	static inline void fastFree(void* ptr) {
		if (ptr) {
			_aligned_free(ptr);
		}
	}



	class Allocator
	{
	public:
		virtual ~Allocator() = 0;
		virtual void* fastMalloc(size_t size) = 0;
		virtual void fastFree(void* ptr) = 0;
	};



	class PoolAllocator : public Allocator
	{
	public:
		PoolAllocator();
		~PoolAllocator();
		void set_size_compare_ratio(float scr);
		void clear();
		virtual void* fastMalloc(size_t size);
		virtual void fastFree(void* ptr);
	private:
		Mutex budgets_lock;
		Mutex payouts_lock;
		unsigned int size_compare_ratio;// 0~256
		std::list< std::pair<size_t, void*> > budgets;
		std::list< std::pair<size_t, void*> > payouts;
	};



	class UnlockedPoolAllocator : public Allocator
	{
	public:
		UnlockedPoolAllocator();
		~UnlockedPoolAllocator();
		void set_size_compare_ratio(float scr);
		void clear();
		virtual void* fastMalloc(size_t size);
		virtual void fastFree(void* ptr);

	private:
		unsigned int size_compare_ratio;// 0~256
		std::list< std::pair<size_t, void*> > budgets;
		std::list< std::pair<size_t, void*> > payouts;
	};



	class VkBufferMemory
	{
	public:
		VkBuffer buffer;
		size_t offset; size_t capacity;
		VkDeviceMemory memory;
		void* mapped_ptr;
		mutable int state;
		int refcount;
	};



	class VkAllocator
	{
	public:
		VkAllocator(const VulkanDevice* _vkdev);
		virtual ~VkAllocator() { clear(); }
		virtual void clear() {}
		virtual VkBufferMemory* fastMalloc(size_t size) = 0;
		virtual void fastFree(VkBufferMemory* ptr) = 0;
		const VulkanDevice* vkdev;
		bool mappable;

	protected:
		VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
		VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
		VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer);
	};



	class VkBlobBufferAllocator : public VkAllocator
	{
	public:
		VkBlobBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkBlobBufferAllocator();
		void set_block_size(size_t size);
		virtual void clear();
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector< std::list< std::pair<size_t, size_t> > > budgets;
		std::vector<VkBufferMemory*> buffer_blocks;
	};



	class VkWeightBufferAllocator : public VkAllocator
	{
	public:
		VkWeightBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkWeightBufferAllocator();
		void set_block_size(size_t block_size);
		virtual void clear();

	public:
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector<size_t> buffer_block_free_spaces;
		std::vector<VkBufferMemory*> buffer_blocks;
		std::vector<VkBufferMemory*> dedicated_buffer_blocks;
	};



	class VkStagingBufferAllocator : public VkAllocator
	{
	public:
		VkStagingBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkStagingBufferAllocator();
		void set_size_compare_ratio(float scr);
		virtual void clear();
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		uint32_t memory_type_index;
		unsigned int size_compare_ratio;
		std::list<VkBufferMemory*> budgets;
	};

	   

	class VkWeightStagingBufferAllocator : public VkAllocator
	{
	public:
		VkWeightStagingBufferAllocator(const VulkanDevice* vkdev);
		virtual ~VkWeightStagingBufferAllocator();
		virtual VkBufferMemory* fastMalloc(size_t size);
		virtual void fastFree(VkBufferMemory* ptr);

	private:
		uint32_t memory_type_index;
	};

};
