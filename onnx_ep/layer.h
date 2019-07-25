#pragma once
#include <vector>
#include "command.h"
#include "tensor.h"

namespace backend {
	class Layer
	{
	public:
		Layer();
		virtual ~Layer();
		virtual int load_param(const std::vector<int>& pd);
		virtual int create_pipeline();
		virtual int destroy_pipeline();

		virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs) const;
		virtual int forward(const Tensor& bottom_blob, Tensor& top_blob) const;
		virtual int forward_inplace(std::vector<Tensor>& bottom_top_blobs) const;
		virtual int forward_inplace(Tensor& bottom_top_blob) const;

		virtual int upload_model(VkTransfer& cmd);

		virtual int forward(const std::vector<DeviceTensor>& bottom_blobs, std::vector<DeviceTensor>& top_blobs, VkCompute& cmd) const;
		virtual int forward(const DeviceTensor& bottom_blob, DeviceTensor& top_blob, VkCompute& cmd) const;
		virtual int forward_inplace(std::vector<DeviceTensor>& bottom_top_blobs, VkCompute& cmd) const;
		virtual int forward_inplace(DeviceTensor& bottom_top_blob, VkCompute& cmd) const;

		const Device* vkdev;
		int typeindex;
		std::vector<int> bottoms;
		std::vector<int> tops;
		bool one_blob_only;
		bool support_inplace;
		bool support_vulkan;
		bool support_packing;
	};
}

