#include "dequantize.h"

namespace backend {
	namespace CPU {
		Dequantize::Dequantize() {
			one_blob_only = true;
			support_inplace = true;
		}

		int Dequantize::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int dims = bottom_top_blob.dims;
			
			if (dims == 1) {
				int w = bottom_top_blob.w;
				const int* intptr = bottom_top_blob;
				float* ptr = bottom_top_blob;
				if (bias_term) {
					if (bias_data_size > 1) {
#pragma omp parallel for num_threads(opt.num_threads)
						for (int i = 0; i < w; i++)
							ptr[i] = intptr[i] * scale + bias_data[i];
					}
					else {
						float bias = bias_data[0];
#pragma omp parallel for num_threads(opt.num_threads)
						for (int i = 0; i < w; i++)
							ptr[i] = intptr[i] * scale + bias;
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < w; i++)
						ptr[i] = intptr[i] * scale;
				}
			}

			if (dims == 2) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				if (bias_term) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < h; i++) {
						const int* intptr = bottom_top_blob.row<const int>(i);
						float* ptr = bottom_top_blob.row(i);
						float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];
						for (int j = 0; j < w; j++)
							ptr[j] = intptr[j] * scale + bias;
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < h; i++) {
						const int* intptr = bottom_top_blob.row<const int>(i);
						float* ptr = bottom_top_blob.row(i);
						for (int j = 0; j < w; j++)
							ptr[j] = intptr[j] * scale;
					}
				}
			}

			if (dims == 3) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				int channels = bottom_top_blob.c;
				int size = w * h;
				if (bias_term) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const int* intptr = bottom_top_blob.channel(q);
						float* ptr = bottom_top_blob.channel(q);
						float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];
						for (int i = 0; i < size; i++)
							ptr[i] = intptr[i] * scale + bias;
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const int* intptr = bottom_top_blob.channel(q);
						float* ptr = bottom_top_blob.channel(q);
						for (int i = 0; i < size; i++)
							ptr[i] = intptr[i] * scale;
					}
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}