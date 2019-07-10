#ifndef LAYER_H
#define LAYER_H

#include "mat.h"
#include "utils.h"
#include "device.h"
#include "command.h"
#include "pipeline.h"

#include <vulkan/vulkan.h>

#define NCNN_MAX_PARAM_COUNT 20

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
		} params[NCNN_MAX_PARAM_COUNT];
	};


}

#endif //!LAYER_H