#include "permute.h"

namespace backend {
	namespace CPU {

		Permute::Permute( ){
			one_blob_only = true;
			support_inplace = false;
		}

		int Permute::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int dims = bottom_blob.dims;

			if (dims == 2) {
				// order_type
				// 0 = w h
				// 1 = h w

				if (order_type == 0)
					top_blob = bottom_blob;
				else if (order_type == 1) {
					top_blob.create(h, w, elemsize, opt.blob_allocator);
					if (top_blob.empty())
						return -100;

					const float* ptr = bottom_blob;
					float* outptr = top_blob;

					for (int i = 0; i < w; i++)	{
						for (int j = 0; j < h; j++)
							outptr[i * h + j] = ptr[j * w + i];
					}
				}

				return 0;
			}

			// order_type
			// 0 = w h c
			// 1 = h w c
			// 2 = w c h
			// 3 = c w h
			// 4 = h c w
			// 5 = c h w

			if (order_type == 0)
				top_blob = bottom_blob;
			else if (order_type == 1) {
				top_blob.create(h, w, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < w; i++) {
						for (int j = 0; j < h; j++)
							outptr[i * h + j] = ptr[j * w + i];						
					}
				}
			}
			else if (order_type == 2) {
				top_blob.create(w, channels, h, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < h; q++) {
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < channels; i++) {
						const float* ptr = bottom_blob.channel(i).row(q);
						for (int j = 0; j < w; j++)
							outptr[i * w + j] = ptr[j];
						
					}
				}
			}
			else if (order_type == 3) {
				top_blob.create(channels, w, h, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < h; q++)	{
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < w; i++)	{
						for (int j = 0; j < channels; j++) {
							const float* ptr = bottom_blob.channel(j).row(q);
							outptr[i * channels + j] = ptr[i];
						}
					}
				}
			}
			else if (order_type == 4) {
				top_blob.create(h, channels, w, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < w; q++) {
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < channels; i++) {
						const float* ptr = bottom_blob.channel(i);
						for (int j = 0; j < h; j++)
							outptr[i * h + j] = ptr[j * w + q];
					}
				}
			}
			else if (order_type == 5) {
				top_blob.create(channels, h, w, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < w; q++)	{
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < channels; j++) {
							const float* ptr = bottom_blob.channel(j);
							outptr[i * channels + j] = ptr[i * w + q];
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