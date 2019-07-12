#include "hardsigmoid.h"

namespace backend {
	namespace CPU {
		HardSigmoid::HardSigmoid() {
			one_blob_only = true;
			support_inplace = true;
		}

		int HardSigmoid::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				float* ptr = bottom_top_blob.channel(q);
				for (int i = 0; i < size; i++) {
					if (ptr[i] < lower)
						ptr[i] = 0.f;
					else if (ptr[i] > upper)
						ptr[i] = 1.f;
					else
						ptr[i] = ptr[i] * alpha + beta;
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}