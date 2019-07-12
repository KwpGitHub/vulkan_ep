#include "reorg.h"

namespace backend {
	namespace CPU {
		Reorg::Reorg() {
			one_blob_only = true;
			support_inplace = false;
		}

		int Reorg::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int outw = w / stride;
			int outh = h / stride;
			int outc = channels * stride * stride;

			top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				const Mat m = bottom_blob.channel(q);
				for (int sh = 0; sh < stride; sh++) {
					for (int sw = 0; sw < stride; sw++) {
						float* outptr = top_blob.channel(q * stride * stride + sh * stride + sw);
						for (int i = 0; i < outh; i++) {
							const float* sptr = m.row(i * stride + sh) + sw;
							for (int j = 0; j < outw; j++) {
								outptr[0] = sptr[0];
								sptr += stride;
								outptr++;
							}
						}
					}
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}