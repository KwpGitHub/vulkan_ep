#include "squeeze.h"

namespace backend {
	namespace CPU {

		Squeeze::Squeeze() {
			one_blob_only = true;
			support_inplace = false;
		}
		
		int Squeeze::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			int dims = bottom_blob.dims;
			top_blob = bottom_blob;

			if (squeeze_c && dims == 3 && channels == 1) {
				if (squeeze_h && h == 1)
					top_blob = bottom_blob.reshape(w, opt.blob_allocator);
				else
					top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
			}
			else if (squeeze_h && dims >= 2 && h == 1) {
				if (squeeze_w && w == 1)
					top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
				else
					top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
			}
			else if (squeeze_w && dims >= 1 && w == 1) {
				if (squeeze_h && h == 1)
					top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
				else
					top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
			}

			if (top_blob.empty())
				return -100;

			return 0;
		}
	}
	namespace GPU {

	}
}
