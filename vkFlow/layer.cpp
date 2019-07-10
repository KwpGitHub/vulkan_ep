#include "layer.h"
namespace backend {

	Layer::Layer()
	{
		one_blob_only = false;
		support_inplace = false;
		support_vulkan = false;
		support_packing = false;

		vkdev = 0;

	}

	Layer::~Layer()	{	}

	int Layer::load_param(const ParamDict& /*pd*/)
	{
		return 0;
	}

	int Layer::create_pipeline(const Option& /*opt*/)
	{
		return 0;
	}

	int Layer::destroy_pipeline(const Option& /*opt*/)
	{
		return 0;
	}


	int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
	{
		if (!support_inplace)
			return -1;

		top_blobs = bottom_blobs;
		for (int i = 0; i < (int)top_blobs.size(); i++)
		{
			top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
			if (top_blobs[i].empty())
				return -100;
		}

		return forward_inplace(top_blobs, opt);
	}

	int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
	{
		if (!support_inplace)
			return -1;

		top_blob = bottom_blob.clone(opt.blob_allocator);
		if (top_blob.empty())
			return -100;

		return forward_inplace(top_blob, opt);
	}

	int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
	{
		return -1;
	}

	int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
	{
		return -1;
	}


	int Layer::upload_model(VkTransfer& /*cmd*/, const Option& /*opt*/)
	{
		return 0;
	}

	int Layer::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
	{
		if (!support_inplace)
			return -1;

		top_blobs.resize(bottom_blobs.size());
		for (int i = 0; i < (int)top_blobs.size(); i++)
		{
			top_blobs[i].create_like(bottom_blobs[i], bottom_blobs[i].allocator, bottom_blobs[i].staging_allocator);
			if (top_blobs[i].empty())
				return -100;

			cmd.record_clone(bottom_blobs[i], top_blobs[i]);
		}

		return forward_inplace(top_blobs, cmd, opt);
	}

	int Layer::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
	{
		if (!support_inplace)
			return -1;

		top_blob.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);
		if (top_blob.empty())
			return -100;

		cmd.record_clone(bottom_blob, top_blob);

		return forward_inplace(top_blob, cmd, opt);
	}

	int Layer::forward_inplace(std::vector<VkMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
	{
		return -1;
	}

	int Layer::forward_inplace(VkMat& /*bottom_top_blob*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
	{
		return -1;
	}

}