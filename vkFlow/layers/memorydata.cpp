#include "memorydata.h"

namespace backend {
	namespace CPU {
		MemoryData::MemoryData() {
			one_blob_only = false;
			support_inplace = false;
		}

		int MemoryData::forward(const std::vector<Mat>& /*bottom_blobs*/, std::vector<Mat>& top_blobs, const Option& opt) const {
			Mat& top_blob = top_blobs[0];
			top_blob = data.clone(opt.blob_allocator);
			if (top_blob.empty())
				return -100;
			return 0;
		}

	}
	namespace GPU {

	}
}