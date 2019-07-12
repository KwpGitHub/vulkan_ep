#include "expanddims.h"

namespace backend {
	namespace CPU {
		ExpandDim::ExpandDim() {
			one_blob_only = true;
			support_inplace = false;
		}

		int ExpandDim::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
		{
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int dims = bottom_blob.dims;
			top_blob = bottom_blob;
			if (dims == 1) {
				if (expand_w) {
					if (expand_h)
						top_blob = bottom_blob.reshape(1, 1, w, opt.blob_allocator);
					else if (expand_c)
						top_blob = bottom_blob.reshape(1, w, 1, opt.blob_allocator);
					else
						top_blob = bottom_blob.reshape(1, w, opt.blob_allocator);
				}
				else if (expand_h) {
					if (expand_c)
						top_blob = bottom_blob.reshape(w, 1, 1, opt.blob_allocator);
					else
						top_blob = bottom_blob.reshape(w, 1, opt.blob_allocator);
				}
			}
			else if (dims == 2) {
				if (expand_w)
					top_blob = bottom_blob.reshape(1, w, h, opt.blob_allocator);
				else if (expand_h)
					top_blob = bottom_blob.reshape(w, 1, h, opt.blob_allocator);
				else if (expand_c)
					top_blob = bottom_blob.reshape(w, h, 1, opt.blob_allocator);
			}

			if (top_blob.empty())
				return -100;

			return 0;
		}
	}
	namespace GPU {

	}
}