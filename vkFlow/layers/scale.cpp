#include "scale.h"

namespace backend {
	namespace CPU {
		Scale::Scale() {
			one_blob_only = true;
			support_inplace = true;
		}

		int Scale::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const {
			Mat& bottom_top_blob = bottom_top_blobs[0];
			const Mat& scale_blob = bottom_top_blobs[1];

			int dims = bottom_top_blob.dims;

			if (dims == 1) {
				int w = bottom_top_blob.w;
				float* ptr = bottom_top_blob;

				if (bias_term) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < w; i++)
						ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < w; i++)
						ptr[i] *= scale_blob[i];
				}
			}

			if (dims == 2) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;

				if (bias_term) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < h; i++)	{
						float* ptr = bottom_top_blob.row(i);
						float s = scale_blob[i];
						float bias = bias_data[i];
						for (int j = 0; j < w; j++)
							ptr[j] = ptr[j] * s + bias;
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < h; i++)	{
						float* ptr = bottom_top_blob.row(i);
						float s = scale_blob[i];
						for (int j = 0; j < w; j++)						
							ptr[j] *= s;
						
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
						float* ptr = bottom_top_blob.channel(q);
						float s = scale_blob[q];
						float bias = bias_data[q];
						for (int i = 0; i < size; i++)
							ptr[i] = ptr[i] * s + bias;
						
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						float* ptr = bottom_top_blob.channel(q);
						float s = scale_blob[q];
						for (int i = 0; i < size; i++)
							ptr[i] *= s;
						
					}
				}
			}

			return 0;
		}

		int Scale::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			std::vector<Mat> bottom_top_blobs(2);
			bottom_top_blobs[0] = bottom_top_blob;
			bottom_top_blobs[1] = scale_data;
			return forward_inplace(bottom_top_blobs, opt);
		}
	}
	namespace GPU {

	}
}