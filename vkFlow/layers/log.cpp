#include "log.h"

namespace backend {
	namespace CPU {
		Log::Log() {
			one_blob_only = true;
			support_inplace = true;
		}


		int Log::forward_inplace(Mat& bottom_top_blob, const Option& opt) const	{
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;
			if (base == -1.f) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++)
						ptr[i] = log(shift + ptr[i] * scale);
				}
			}
			else {
				float log_base_inv = 1.f / log(base);
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++)
						ptr[i] = log(shift + ptr[i] * scale) * log_base_inv;
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}