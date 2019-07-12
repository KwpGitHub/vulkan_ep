#include "relu.h"

namespace backend {
	namespace CPU {
		ReLU::ReLU() {
			one_blob_only = true;
			support_inplace = true;
		}

		int ReLU::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
		{
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

			if (slope == 0.f) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					signed char* ptr = bottom_top_blob.channel(q); 
					for (int i = 0; i < size; i++) {
						if (ptr[i] < 0)
							ptr[i] = 0;
					}
				}
			}
			else
			{
				// TODO
				// #pragma omp parallel for num_threads(opt.num_threads)
				// for (int q=0; q<channels; q++)
				// {
				//     float* ptr = bottom_top_blob.channel(q);

				//     for (int i=0; i<size; i++)
				//     {
				//         if (ptr[i] < 0)
				//             ptr[i] *= slope;
				//     }
				// }
			}

			return 0;
		}

		int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
		{
			if (bottom_top_blob.elemsize == 1u)
				return ReLU::forward_inplace_int8(bottom_top_blob, opt);
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

			if (slope == 0.f) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++) {
						if (ptr[i] < 0)
							ptr[i] = 0;
					}
				}
			}
			else {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++) {
						if (ptr[i] < 0)
							ptr[i] *= slope;
					}
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}