#ifndef TENSOR_H
#define TENSOR_H

#include <intrin.h>
#include <pybind11/numpy.h>
#define VK_EP_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

#include <vector>
#include "allocator.h"

namespace backend {

	class DeviceTensor;

	// the three dimension matrix
	class Tensor {
	public:
		/*Tensor();
		//Tensor(int w, size_t elemsize = 4u, CPU_Allocator * allocator = 0);
		~Tensor();
		inline Tensor& operator=(const Tensor& m);
		void fill(float v);
		void fill(int v);
		template <typename T> void fill(T v);
		Tensor clone(CPU_Allocator* allocator = 0) const;
		Tensor reshape(int w, CPU_Allocator* allocator = 0) const;
		Tensor reshape(int w, int h, CPU_Allocator* allocator = 0) const;
		Tensor reshape(int w, int h, int c, CPU_Allocator* allocator = 0) const;
		void create_like(const Tensor& m, CPU_Allocator* allocator = 0);
		void create_like(const DeviceTensor& m, CPU_Allocator* allocator = 0);

		void addref();
		void release();

		bool empty() const;
		size_t total() const;

		Tensor channel(int c);
		const Tensor channel(int c) const;
		float* row(int y);
		const float* row(int y) const;
		template<typename T> T* row(int y);
		template<typename T> const T* row(int y) const;

		Tensor channel_range(int c, int channels);
		const Tensor channel_range(int c, int channels) const;
		Tensor row_range(int y, int rows);
		const Tensor row_range(int y, int rows) const;
		Tensor range(int x, int n);
		const Tensor range(int x, int n) const;

		template<typename T> operator T* ();
		template<typename T> operator const T* () const;

		float& operator[](int i);
		const float& operator[](int i) const;*/

#if VK_EP_PIXEL
		enum
		{
			PIXEL_CONVERT_SHIFT = 16,
			PIXEL_FORMAT_MASK = 0x0000ffff,
			PIXEL_CONVERT_MASK = 0xffff0000,

			PIXEL_RGB = 1,
			PIXEL_BGR = (1 << 1),
			PIXEL_GRAY = (1 << 2),
			PIXEL_RGBA = (1 << 3),

			PIXEL_RGB2BGR = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
			PIXEL_RGB2GRAY = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

			PIXEL_BGR2RGB = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
			PIXEL_BGR2GRAY = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

			PIXEL_GRAY2RGB = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
			PIXEL_GRAY2BGR = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),

			PIXEL_RGBA2RGB = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
			PIXEL_RGBA2BGR = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
			PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
		};
		// convenient construct from pixel data
		static Tensor from_pixels(const unsigned char* pixels, int type, int w, int h, CPU_Allocator* allocator = 0);
		// convenient construct from pixel data and resize to specific size
		static Tensor from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, CPU_Allocator* allocator = 0);

		// convenient export to pixel data
		void to_pixels(unsigned char* pixels, int type) const;
		// convenient export to pixel data and resize to specific size
		void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
#endif // VK_EP_PIXEL

		//void substract_mean_normalize(const float* mean_vals, const float* norm_vals);
		//static Tensor from_float16(const unsigned short* data, int size);

		void* data;
		int* refcount;

		// element size in bytes
		// 4 = float32/int32
		// 2 = float16
		// 1 = int8/uint8
		// 0 = empty
		size_t elemsize = 4u;

		// packed count inside element
		// c/1-h-w-1  h/1-w-1  w/1-1  scalar
		// c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
		// c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
		int packing;

		CPU_Allocator* allocator;
		
		std::vector<int> dim;
		int dims;
		size_t cstep;
	};


	// the three dimension matrix, vulkan version
	class DeviceTensor {
	public:
	/*	DeviceTensor();
		DeviceTensor(const DeviceTensor& m);
		~DeviceTensor();
		DeviceTensor& operator=(const DeviceTensor& m);
	
		void create_like(const Tensor& m, Allocator* allocator, Allocator* staging_allocator);
		void create_like(const DeviceTensor& m, Allocator* allocator, Allocator* staging_allocator);

		void prepare_staging_buffer();
		void discard_staging_buffer();

		void upload(const Tensor& m);
		void download(Tensor& m) const;

		Tensor mapped() const;
		void* mapped_ptr() const;

		void addref();
		void release();

		bool empty() const;
		size_t total() const;

		DeviceTensor channel(int c);
		const DeviceTensor channel(int c) const;

		DeviceTensor channel_range(int c, int channels);
		const DeviceTensor channel_range(int c, int channels) const;
		DeviceTensor row_range(int y, int rows);
		const DeviceTensor row_range(int y, int rows) const;
		DeviceTensor range(int x, int n);
		const DeviceTensor range(int x, int n) const;

		VkBuffer buffer() const;
		size_t buffer_offset() const;
		VkBuffer staging_buffer() const;
		size_t staging_buffer_offset() const;*/

		BufferMemory* data;
		size_t offset;

		BufferMemory* staging_data;

		int* refcount;
		int* staging_refcount;

		// element size in bytes
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

		std::vector<int> dim;
		int dims;
		size_t cstep;
	};
}

namespace backend {

}



#endif //!TENSOR_H