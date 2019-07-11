#ifndef OPTIONS_H
#define OPTIONS_H

#include "utils.h"

namespace backend {

	class VkAllocator;
	class Allocator;

	class Option

	{
	public:
		// default option
		Option();

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

#endif //!OPTIONS_H