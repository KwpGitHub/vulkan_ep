#include "bias.h"

namespace backend {
	namespace CPU {
		Bias::Bias(){
			one_blob_only = true;
			support_inplace = true;
		}

		int Bias::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				float* ptr = bottom_top_blob.channel(q);
				float bias = bias_data[q];

				for (int i = 0; i < size; i++)
					ptr[i] += bias;
			}

			return 0;
		}
	}
	namespace GPU {

	}
}