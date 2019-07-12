#include "normalize.h"

namespace backend {
	namespace CPU {
		Normalize::Normalize() {
			one_blob_only = true;
			support_inplace = false;
		}


		int Normalize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int size = w * h;

			top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			if (across_spatial && across_channel) {
				Mat square_sum_blob;
				square_sum_blob.create(channels, elemsize, opt.workspace_allocator);
				if (square_sum_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					float ssum = 0.f;
					for (int i = 0; i < size; i++)
						ssum += ptr[i] * ptr[i];
					square_sum_blob[q] = ssum;
				}
				float ssum = eps;
				for (int q = 0; q < channels; q++)
					ssum += square_sum_blob[q];
				float a = 1.f / sqrt(ssum);
				if (channel_shared) {
					float scale = a * scale_data[0];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++)
							outptr[i] = ptr[i] * scale;
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float* outptr = top_blob.channel(q);
						float scale = a * scale_data[q];
						for (int i = 0; i < size; i++)
							outptr[i] = ptr[i] * scale;
					}
				}

				return 0;
			}

			if (across_spatial && !across_channel) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					float* outptr = top_blob.channel(q);
					float ssum = eps;
					for (int i = 0; i < size; i++)
						ssum += ptr[i] * ptr[i];
					float a = 1.f / sqrt(ssum);
					float scale = a * (channel_shared ? scale_data[0] : scale_data[q]);
					for (int i = 0; i < size; i++)
						outptr[i] = ptr[i] * scale;
				}

				return 0;
			}

			if (!across_spatial && across_channel) {
				// square sum, 1 / sqrt(ssum)
				Mat square_sum_blob;
				square_sum_blob.create(size, elemsize, opt.workspace_allocator);
				if (square_sum_blob.empty())
					return -100;
				if (channel_shared) {
					float scale = scale_data[0];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < size; i++) {
						float ssum = eps;
						for (int q = 0; q < channels; q++) {
							const float* ptr = bottom_blob.channel(q);
							ssum += ptr[i] * ptr[i];
						}
						square_sum_blob[i] = 1.f / sqrt(ssum) * scale;
					}

#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++) 
							outptr[i] = ptr[i] * square_sum_blob[i];						
					}
				}
				else {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int i = 0; i < size; i++) {
						float ssum = eps;
						for (int q = 0; q < channels; q++) {
							const float* ptr = bottom_blob.channel(q);
							ssum += ptr[i] * ptr[i];
						}
						square_sum_blob[i] = 1.f / sqrt(ssum);
					}

#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float* outptr = top_blob.channel(q);
						float scale = scale_data[q];
						for (int i = 0; i < size; i++)
							outptr[i] = ptr[i] * square_sum_blob[i] * scale;
					}
				}

				return 0;
			}

			return 0;
		}
	}
	namespace GPU {

	}
}