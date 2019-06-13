#pragma once

#include "../device.h"

#include <vulkan/vulkan.hpp>

namespace pipeline{
namespace arr {
	auto copyBuf(pipeline::Device& device
	             , vk::Buffer src, vk::Buffer dst
	             , size_t size_bytes
	             , size_t src_offset=0
	             , size_t dst_offset=0
	             )-> void;
} // namespace arr
} // namespace pipeline
