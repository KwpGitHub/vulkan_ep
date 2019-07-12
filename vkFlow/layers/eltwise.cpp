#include "eltwise.h"
#include <algorithm>
namespace backend {
	namespace CPU {
		Eltwise::Eltwise() {
			one_blob_only = false;
			support_inplace = false;// TODO inplace reduction
		}
	

		int Eltwise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
			const Mat& bottom_blob = bottom_blobs[0];
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int size = w * h;
			Mat& top_blob = top_blobs[0];
			top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			if (op_type == Operation_PROD) {
				const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					const float* ptr1 = bottom_blob1.channel(q);
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < size; i++)
						outptr[i] = ptr[i] * ptr1[i];
				}

				for (size_t b = 2; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob1.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++)
							outptr[i] *= ptr[i];
					}
				}
			}
			else if (op_type == Operation_SUM) {
				if (coeffs.w == 0) {
					const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						const float* ptr1 = bottom_blob1.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++)
							outptr[i] = ptr[i] + ptr1[i];						
					}

					for (size_t b = 2; b < bottom_blobs.size(); b++) {
						const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
						for (int q = 0; q < channels; q++) {
							const float* ptr = bottom_blob1.channel(q);
							float* outptr = top_blob.channel(q);
							for (int i = 0; i < size; i++)
								outptr[i] += ptr[i];
						}
					}
				}
				else {
					// first blob
					const Mat& bottom_blob1 = bottom_blobs[1];
					float coeff0 = coeffs[0];
					float coeff1 = coeffs[1];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						const float* ptr1 = bottom_blob1.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++)
							outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
					}

					for (size_t b = 2; b < bottom_blobs.size(); b++) {
						const Mat& bottom_blob1 = bottom_blobs[b];
						float coeff = coeffs[b];
#pragma omp parallel for num_threads(opt.num_threads)
						for (int q = 0; q < channels; q++) {
							const float* ptr = bottom_blob1.channel(q);
							float* outptr = top_blob.channel(q);
							for (int i = 0; i < size; i++)
								outptr[i] += ptr[i] * coeff;
						}
					}
				}
			}
			else if (op_type == Operation_MAX)	{
				// first blob
				const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					const float* ptr1 = bottom_blob1.channel(q);
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < size; i++)
						outptr[i] = std::max<float>(ptr[i], ptr1[i]);
				}

				for (size_t b = 2; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob1.channel(q);
						float* outptr = top_blob.channel(q);
						for (int i = 0; i < size; i++)
							outptr[i] = std::max<float>(outptr[i], ptr[i]);
					}
				}
			}

			return 0;
		}


	}
	namespace GPU {

	}
}