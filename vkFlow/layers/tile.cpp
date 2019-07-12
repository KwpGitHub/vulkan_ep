#include "tile.h"

namespace backend {
	namespace CPU {
		Tile::Tile()
		{
			one_blob_only = true;
			support_inplace = false;
		}

		int Tile::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
		{
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;

			if (dim == 0) {
				top_blob.create(w, h, channels * tiles, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				const float* ptr = bottom_blob;
				int size = bottom_blob.cstep * channels;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int p = 0; p < tiles; p++)	{
					float* outptr = top_blob.channel(p * channels);

					for (int i = 0; i < size; i++) {
						outptr[i] = ptr[i];
					}
				}
			}
			else if (dim == 1) {
				top_blob.create(w, h * tiles, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					float* outptr = top_blob.channel(q);
					for (int p = 0; p < tiles; p++) {
						for (int i = 0; i < size; i++) {
							outptr[i] = ptr[i];
						}

						outptr += size;
					}
				}
			}
			else if (dim == 2)
			{
				top_blob.create(w * tiles, h, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++)
				{
					const float* ptr = bottom_blob.channel(q);
					float* outptr = top_blob.channel(q);

					for (int i = 0; i < h; i++)	{
						for (int p = 0; p < tiles; p++)	{
							for (int j = 0; j < w; j++)	{
								outptr[j] = ptr[j];
							}

							outptr += w;
						}

						ptr += w;
					}
				}
			}

			return 0;
		}

	}
	namespace GPU {

	}
}