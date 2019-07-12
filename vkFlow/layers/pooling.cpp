#include "pooling.h"
#include <algorithm>

namespace backend {
	namespace CPU {
		Pooling::Pooling() {
			one_blob_only = true;
			support_inplace = false;
		}

		int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;

			if (global_pooling) {
				top_blob.create(channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				int size = w * h;

				if (pooling_type == PoolMethod_MAX) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float max = ptr[0];
						for (int i = 0; i < size; i++)
							max = std::max<float>(max, ptr[i]);
						top_blob[q] = max;
					}
				}
				else if (pooling_type == PoolMethod_AVE) {
#pragma omp parallel for num_threads(opt.num_threads)
					for (int q = 0; q < channels; q++) {
						const float* ptr = bottom_blob.channel(q);
						float sum = 0.f;
						for (int i = 0; i < size; i++)
							sum += ptr[i];
						top_blob[q] = sum / size;
					}
				}

				return 0;
			}

			Mat bottom_blob_bordered = bottom_blob;

			float pad_value = 0.f;
			if (pooling_type == PoolMethod_MAX)
				pad_value = -FLT_MAX;
			else if (pooling_type == PoolMethod_AVE)
				pad_value = 0.f;

			int wtailpad = 0;
			int htailpad = 0;

			if (pad_mode == 0) {
				int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
				int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

				if (wtail != 0)
					wtailpad = stride_w - wtail;
				if (htail != 0)
					htailpad = stride_h - htail;

				copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
				if (bottom_blob_bordered.empty())
					return -100;

				w = bottom_blob_bordered.w;
				h = bottom_blob_bordered.h;
			}
			else if (pad_mode == 1) {
				copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
				if (bottom_blob_bordered.empty())
					return -100;

				w = bottom_blob_bordered.w;
				h = bottom_blob_bordered.h;
			}
			else if (pad_mode == 2) {
				int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
				int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
				if (wpad > 0 || hpad > 0) {
					copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
					if (bottom_blob_bordered.empty())
						return -100;
				}

				w = bottom_blob_bordered.w;
				h = bottom_blob_bordered.h;
			}

			int outw = (w - kernel_w) / stride_w + 1;
			int outh = (h - kernel_h) / stride_h + 1;

			top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			const int maxk = kernel_w * kernel_h;
			std::vector<int> _space_ofs(maxk);

			int* space_ofs = &_space_ofs[0];
			{
				int p1 = 0;
				int p2 = 0;
				int gap = w - kernel_w;
				for (int i = 0; i < kernel_h; i++) {
					for (int j = 0; j < kernel_w; j++) {
						space_ofs[p1] = p2;
						p1++;
						p2++;
					}
					p2 += gap;
				}
			}

			if (pooling_type == PoolMethod_MAX) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const Mat m = bottom_blob_bordered.channel(q);
					float* outptr = top_blob.channel(q);

					for (int i = 0; i < outh; i++) {
						for (int j = 0; j < outw; j++) {
							const float* sptr = m.row(i * stride_h) + j * stride_w;
							float max = sptr[0];
							for (int k = 0; k < maxk; k++) {
								float val = sptr[space_ofs[k]];
								max = std::max<float>(max, val);
							}

							outptr[j] = max;
						}

						outptr += outw;
					}
				}
			}
			else if (pooling_type == PoolMethod_AVE) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const Mat m = bottom_blob_bordered.channel(q);
					float* outptr = top_blob.channel(q);

					for (int i = 0; i < outh; i++) { 
						for (int j = 0; j < outw; j++) {
							const float* sptr = m.row(i * stride_h) + j * stride_w;
							float sum = 0;
							for (int k = 0; k < maxk; k++) {
								float val = sptr[space_ofs[k]];
								sum += val;
							}
							outptr[j] = sum / maxk;
						}
						outptr += outw;
					}

					// fix pad
					if (pad_top != 0) {
						const float scale = (float)kernel_h / (kernel_h - pad_top);
						outptr = top_blob.channel(q).row(0);
						for (int i = 0; i < outw; i++) 
							outptr[i] *= scale;
					}
					if (pad_bottom + htailpad != 0) {
						const float scale = (float)kernel_h / (kernel_h - pad_bottom - htailpad);
						outptr = top_blob.channel(q).row(outh - 1);
						for (int i = 0; i < outw; i++) 
							outptr[i] *= scale;
					}
					if (pad_left != 0) {
						const float scale = (float)kernel_w / (kernel_w - pad_left);
						outptr = top_blob.channel(q);
						for (int i = 0; i < outh; i++) {
							*outptr *= scale;
							outptr += outw;
						}
					}
					if (pad_right + wtailpad != 0) {
						const float scale = (float)kernel_w / (kernel_w - pad_right - wtailpad);
						outptr = top_blob.channel(q);
						outptr += outw - 1;
						for (int i = 0; i < outh; i++) { 
							*outptr *= scale;
							outptr += outw;
						}
					}
				}
			}

			return 0;
		}

	}
	namespace GPU {

	}
}