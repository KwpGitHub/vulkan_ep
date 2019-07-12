#include "split.h"

namespace backend {
	namespace CPU {
		Split::Split() {
			one_blob_only = false;
			support_inplace = false;
			support_vulkan = true;
		}


		int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const {
			const Mat& bottom_blob = bottom_blobs[0];
			for (size_t i = 0; i < top_blobs.size(); i++)
				top_blobs[i] = bottom_blob;

			return 0;
		}

		int Split::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& /*cmd*/, const Option& /*opt*/) const
		{
			const VkMat& bottom_blob = bottom_blobs[0];
			for (size_t i = 0; i < top_blobs.size(); i++)
				top_blobs[i] = bottom_blob;

			return 0;
		}

	}
	namespace GPU {

	}
}