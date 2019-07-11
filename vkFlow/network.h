#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <vector>
#include "utils.h"
#include "layer.h"
#include "mat.h"

namespace backend {

	class VkCompute;
	class Network
	{
	public:
		Network();
		~Network();

		Option opt;
		void set_vulkan_device(int device_index);
		void set_vulkan_device(const VulkanDevice* vkdev);

		void clear();

	protected:
		int fuse_network();
		int upload_model();
		int create_pipeline();
		int destry_pipeline();

		int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, Option& opt) const;
#
		std::vector<Layer*> layers;
		const VulkanDevice* vkdev;

		VkAllocator* weight_vkallocator;
		VkAllocator* weight_staging_vkallocator;

		backend::Layer* cast_float32_to_float16;
		backend::Layer* cast_float16_to_float32;
		backend::Layer* packing_pack1;
		backend::Layer* packing_pack4;

	};
}


#endif //!NET_H