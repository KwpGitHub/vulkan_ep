#ifndef TENSOR_H
#define TENSOR_H

#include <intrin.h>
#define VK_EP_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

#include<vector>
#include "allocator.h"

namespace backend {
	class Tensor
	{
		Tensor();
	
		BufferMemory* data;
		BufferMemory* staging_data;
		
		size_t offset;
		int* refcount;
		int* staging_refcount;

		// 4 = float32/int32
		// 2 = float16
		// 1 = int8/uint8
		// 0 = empty
		size_t elemsize;

		// packed count inside element
		// c/1-h-w-1  h/1-w-1  w/1-1  scalar
		// c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
		// c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
		int packing;

		Allocator* allocator;
		Allocator* staging_allocator;
		std::vector<int> dims;

		Tensor& operator=(const Tensor& m);

		void prepare_staging_buffer();
		void discard_staging_buffer();

		void upload(const std::vector<float>& m) { memcpy(mapped_ptr(), m.data, m.size() * sizeof(float)); }
		void download(const std::vector<float>& m) const { memcpy(m.data, mapped_ptr(), total() * elemsize); }

		Tensor mapped() const; //cpu tensor

		void* mapped_ptr() const {
			BufferMemory* mappable_data = allocator->mappable ? data : staging_data;
			return (unsigned char*)mappable_data->mapped_ptr + mappable_data->offset + offset;
		}

		void addref() {
			if (refcount) VK_EP_XADD(refcount, 1);
			if (staging_refcount) VK_EP_XADD(staging_refcount, 1);
		}
		// refcount--
		void release();

		bool empty() const;
		size_t total() const;

		VkBuffer buffer() const;
		size_t buffer_offset() const;
		VkBuffer staging_buffer() const;
		size_t staging_buffer_offset() const;

	};
}

#endif //!TENSOR_H