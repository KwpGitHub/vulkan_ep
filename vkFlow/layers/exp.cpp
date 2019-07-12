#include "exp.h"

namespace backend {
	namespace CPU {
		Exp::Exp() {
			one_blob_only = true;
			support_inplace = true;
		}

		int Exp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;
			if (base == -1.f) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++)
						ptr[i] = exp(shift + ptr[i] * scale);
				}
			}
			else {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++)
						ptr[i] = pow(base, (shift + ptr[i] * scale));
				}
			}

			return 0;
		}



	}
	namespace GPU {

	}
}