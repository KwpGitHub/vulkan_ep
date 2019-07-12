#include "flatten.h"

namespace backend {
	namespace CPU {
		Flatten::Flatten() {
			one_blob_only = true;
			support_inplace = false;
		}

		int Flatten::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int size = w * h;
			top_blob.create(size * channels, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				const float* ptr = bottom_blob.channel(q);
				float* outptr = (float*)top_blob + size * q;
				for (int i = 0; i < size; i++)
					outptr[i] = ptr[i];
			}

			return 0;
		}
	}
	namespace GPU {

	}
}