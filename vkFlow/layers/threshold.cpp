#include "threshold.h"

namespace backend {
	namespace CPU {
		Threshold::Threshold()
		{
			one_blob_only = true;
			support_inplace = true;
		}

		int Threshold::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
		{
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				float* ptr = bottom_top_blob.channel(q);
				for (int i = 0; i < size; i++)
					ptr[i] = ptr[i] > threshold ? 1.f : 0.f;
			}

			return 0;
		}

	}
	namespace GPU {

	}
}