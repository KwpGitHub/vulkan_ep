#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#define MALLOC_ALIGN 20
#include <intrin.h>
#define VK_EP_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

#include <Windows.h>
#include <cstdlib>
#include <list>
#include <vector>
#include <vulkan/vulkan.h>
#include "device.h"

namespace backend {
	template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
	{
		return (_Tp*)(((size_t)ptr + n - 1) & -n);
	}
	static inline size_t alignSize(size_t sz, int n)
	{
		return (sz + n - 1) & -n;
	}
	static inline void* fastMalloc(size_t size)
	{
		return _aligned_malloc(size, MALLOC_ALIGN);
	}
	static inline void fastfree(void* ptr) {
		if (ptr)
			_aligned_free(ptr);
	}


	class CPU_Allocator
	{
	public:
		virtual ~CPU_Allocator() = 0;
		virtual void* fastMalloc(size_t size) = 0;
		virtual void fastFree(void* ptr) = 0;
	};

	class BufferMemory {
	public:
		VkBuffer buffer;
		size_t offset;
		size_t capacity;
		VkDeviceMemory memory;
		void* mapped_ptr;
		;
		/*
		    // 0=null
			// 1=created
			// 2=transfer
			// 3=compute
			// 4=readonly
		*/
		mutable int state; 
		int refcount;
	};


	class Allocator
	{
	public: 
		Allocator(const Device* _dev);
		virtual ~Allocator(){ clear(); }
		virtual void clear() {}
		virtual BufferMemory* fastMalloc(size_t size) = 0;
		virtual void fastFree(BufferMemory* ptr) = 0;

		const Device* dev;
		bool mappable;

	protected:
		VkBuffer create_buffer(size_t, VkBufferUsageFlags usage);
		VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
		VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer);
	};


	class BlobBufferAllocator : public Allocator {
	public:
		BlobBufferAllocator(const Device* dev);
		virtual ~BlobBufferAllocator();
		void set_block_size(size_t size);
		virtual void clear();
		virtual BufferMemory* fastMalloc(size_t size);
		virtual void fastFree(BufferMemory* ptr);

	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector< std::list< std::pair<size_t, size_t> > > budgets;
		std::vector<BufferMemory*> buffer_blocks;
	};


	class WeightBufferAllocator : public Allocator {
	public:
		WeightBufferAllocator(const Device* dev);
		virtual ~WeightBufferAllocator();
		void set_block_size(size_t block_size);
		virtual void clear();
		virtual BufferMemory* fastMalloc(size_t size);
		virtual void fastFree(BufferMemory* ptr);
	private:
		size_t block_size;
		size_t buffer_offset_alignment;
		std::vector<size_t> buffer_block_free_spaces;
		std::vector<BufferMemory*> buffer_blocks;
		std::vector<BufferMemory*> dedicated_buffer_blocks;
	};


	class StagingBufferAllocator : public Allocator
	{
	public:
		StagingBufferAllocator(const Device* dev);
		virtual ~StagingBufferAllocator();
		void set_size_compare_ratio(float scr);
		virtual void clear();
		virtual BufferMemory* fastMalloc(size_t size);
		virtual void fastFree(BufferMemory* ptr);

	private:
		uint32_t memory_type_index;
		unsigned int size_compare_ratio;// 0~256
		std::list<BufferMemory*> budgets;
	};


	class WeightStagingBufferAllocator : public Allocator
	{
	public:
		WeightStagingBufferAllocator(const Device* dev);
		virtual ~WeightStagingBufferAllocator();
		virtual BufferMemory* fastMalloc(size_t size);
		virtual void fastFree(BufferMemory* ptr);

	private:
		uint32_t memory_type_index;
	};

}
#endif //!ALLOCATOR_H