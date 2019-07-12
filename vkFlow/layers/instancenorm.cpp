#include "instancenorm.h"

namespace backend {
	namespace CPU {
		InstanceNorm::InstanceNorm() {
			one_blob_only = true;
			support_inplace = true;
		}
		

		int InstanceNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				float* ptr = bottom_top_blob.channel(q);
				float sum = 0.f;
				float sqsum = 0.f;
				for (int i = 0; i < size; i++)
					sum += ptr[i];
				float mean = sum / size;
				float tmp = 0.f;
				for (int i = 0; i < size; i++) {
					tmp = ptr[i] - mean;
					sqsum += tmp * tmp;
				}
				float var = sqsum / size;
				float gamma = gamma_data[q];
				float beta = beta_data[q];
				float a = gamma / (sqrt(var + eps));
				float b = -mean * a + beta;
				for (int i = 0; i < size; i++)
					ptr[i] = ptr[i] * a + b;
			}

			return 0;
		}
	}
	namespace GPU {

	}
}