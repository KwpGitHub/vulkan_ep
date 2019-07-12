#include "concat.h"

namespace backend {
	namespace CPU {
		Concat::Concat() {
			one_blob_only = false;
			support_inplace = false;
		}

		int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const{
			int dims = bottom_blobs[0].dims;
			size_t elemsize = bottom_blobs[0].elemsize;

			if (dims == 1) {
				int top_w = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					top_w += bottom_blob.w;
				}

				Mat& top_blob = top_blobs[0];
				top_blob.create(top_w, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				float* outptr = top_blob;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					int w = bottom_blob.w;
					const float* ptr = bottom_blob;
					memcpy(outptr, ptr, w * elemsize);
					outptr += w;
				}

				return 0;
			}

			if (dims == 2 && axis == 0)	{
				int w = bottom_blobs[0].w;
				int top_h = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					top_h += bottom_blob.h;
				}

				Mat& top_blob = top_blobs[0];
				top_blob.create(w, top_h, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				float* outptr = top_blob;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					int size = w * bottom_blob.h;
					const float* ptr = bottom_blob;
					memcpy(outptr, ptr, size * elemsize);
					outptr += size;
				}

				return 0;
			}

			if (dims == 2 && axis == 1) {
				int h = bottom_blobs[0].h;
				int top_w = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					top_w += bottom_blob.w;
				}

				Mat& top_blob = top_blobs[0];
				top_blob.create(top_w, h, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int i = 0; i < h; i++)
				{
					float* outptr = top_blob.row(i);
					for (size_t b = 0; b < bottom_blobs.size(); b++)
					{
						const Mat& bottom_blob = bottom_blobs[b];

						const float* ptr = bottom_blob.row(i);
						memcpy(outptr, ptr, bottom_blob.w * elemsize);

						outptr += bottom_blob.w;
					}
				}

				return 0;
			}

			if (dims == 3 && axis == 0)
			{
				// concat dim
				int w = bottom_blobs[0].w;
				int h = bottom_blobs[0].h;

				// total channels
				int top_channels = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++)
				{
					const Mat& bottom_blob = bottom_blobs[b];
					top_channels += bottom_blob.c;
				}

				Mat& top_blob = top_blobs[0];
				top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				int q = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++)
				{
					const Mat& bottom_blob = bottom_blobs[b];

					int channels = bottom_blob.c;
					int size = bottom_blob.cstep * channels;

					const float* ptr = bottom_blob;
					float* outptr = top_blob.channel(q);
					memcpy(outptr, ptr, size * elemsize);

					q += channels;
				}

				return 0;
			}

			if (dims == 3 && axis == 1)
			{
				// interleave dim height
				int w = bottom_blobs[0].w;
				int channels = bottom_blobs[0].c;

				// total height
				int top_h = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++)
				{
					const Mat& bottom_blob = bottom_blobs[b];
					top_h += bottom_blob.h;
				}

				Mat& top_blob = top_blobs[0];
				top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* outptr = top_blob.channel(q);
					for (size_t b = 0; b < bottom_blobs.size(); b++) {
						const Mat& bottom_blob = bottom_blobs[b];
						int size = bottom_blob.w * bottom_blob.h;
						const float* ptr = bottom_blob.channel(q);
						memcpy(outptr, ptr, size * elemsize);
						outptr += size;
					}
				}

				return 0;
			}

			if (dims == 3 && axis == 2) {
				int h = bottom_blobs[0].h;
				int channels = bottom_blobs[0].c;
				int top_w = 0;
				for (size_t b = 0; b < bottom_blobs.size(); b++) {
					const Mat& bottom_blob = bottom_blobs[b];
					top_w += bottom_blob.w;
				}
				Mat& top_blob = top_blobs[0];
				top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < h; i++) {
						for (size_t b = 0; b < bottom_blobs.size(); b++) {
							const Mat& bottom_blob = bottom_blobs[b];
							const float* ptr = bottom_blob.channel(q).row(i);
							memcpy(outptr, ptr, bottom_blob.w * elemsize);
							outptr += bottom_blob.w;
						}
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