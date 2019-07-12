#include "selu.h"

namespace backend {
	namespace CPU {
		SeLU::SeLU() {
			one_blob_only = true;
			support_inplace = true;
		}

		int SeLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;
			float alphaxlambda = alpha * lambda;

#pragma omp parallel for num_threads(opt.num_threds)
			for (int q = 0; q < channels; ++q) {
				float* ptr = bottom_top_blob.channel(q);
				for (int i = 0; i < size; ++i) 
					ptr[i] = ptr[i] < 0.f ? (exp(ptr[i]) - 1.f) * alphaxlambda : lambda;
			}

			return 0;
		}
	}
	namespace GPU {

	}
}