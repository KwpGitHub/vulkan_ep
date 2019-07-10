#ifndef LAYER_H
#define LAYER_H
#include <vulkan/vulkan.h>

#include "mat.h"
#include "utils.hpp"
#include "command.h"
#include "pipeline.h"

namespace backend {
	class Layer
	{
	public:
		Layer();
		virtual ~Layer();
		virtual int load_param(const ParamDict& pd);
		virtual int create_pipeline(const Option& opt = Option());
		virtual int destroy_pipeline(const Option& opt = Option());
		
		bool one_blob_only;
		bool support_inplace;
		bool support_vulkan;
		bool support_packing;

		virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt = Option()) const;
		virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt = Option()) const;
		virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt = Option()) const;
		virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt = Option()) const;

		virtual int upload_model(VkTransfer& cmd, const Option& opt = Option());

		virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
		virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt = Option()) const;
		virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
		virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt = Option()) const;

		const VulkanDevice* vkdev;
		int typeindex;
		std::vector<int> bottoms;
		std::vector<int> tops;
	};

}

#endif //!LAYER_H