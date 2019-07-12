#include "batchnorm.h"
#include <math.h>

namespace backend {
	namespace CPU {

		BatchNorm::BatchNorm() {
			one_blob_only = true;
			support_inplace = true;
		}

		int BatchNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int dims = bottom_top_blob.dims;

			if (dims == 1) {
				int w = bottom_top_blob.w;
				float* ptr = bottom_top_blob;
#pragma omp parallel for num_threads(opt.num_threads)
				for (int i = 0; i < w; i++)
					ptr[i] = b_data[i] * ptr[i] + a_data[i];
			}

			if (dims == 2) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
#pragma omp parallel for num_threads(opt.num_threads)
				for (int i = 0; i < h; i++)
				{
					float* ptr = bottom_top_blob.row(i);
					float a = a_data[i];
					float b = b_data[i];
					for (int j = 0; j < w; j++)
						ptr[j] = b * ptr[j] + a;

				}
			}

			if (dims == 3) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++)
				{
					float* ptr = bottom_top_blob.channel(q);
					float a = a_data[q];
					float b = b_data[q];

					for (int i = 0; i < size; i++)
						ptr[i] = b * ptr[i] + a;

				}
			}
			return 0;
		}

	}
	namespace GPU {

	}
}
