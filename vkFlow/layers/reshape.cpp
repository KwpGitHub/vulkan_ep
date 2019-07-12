#include "reshape.h"

namespace backend {
	namespace CPU {
		Reshape::Reshape() {
			one_blob_only = true;
			support_inplace = false;
		}


		int Reshape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			size_t elemsize = bottom_blob.elemsize;
			int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

			if (ndim == 1)	{
				int _w = w;
				if (_w == 0) _w = bottom_blob.w;
				if (_w == -1) _w = total;

				if (permute == 1) {
					top_blob.create(_w, elemsize, opt.blob_allocator);
					if (top_blob.empty())
						return -100;

					// c-h-w to h-w-c
					float* ptr = top_blob;
					for (int i = 0; i < bottom_blob.h; i++) {
						for (int j = 0; j < bottom_blob.w; j++) {
							for (int p = 0; p < bottom_blob.c; p++) {
								const float* bptr = bottom_blob.channel(p);
								*ptr++ = bptr[i * bottom_blob.w + j];
							}
						}
					}
				}
				else {
					top_blob = bottom_blob.reshape(_w, opt.blob_allocator);
				}
			}
			else if (ndim == 2)	{
				int _w = w;
				int _h = h;
				if (_w == 0) _w = bottom_blob.w;
				if (_h == 0) _h = bottom_blob.h;
				if (_w == -1) _w = total / _h;
				if (_h == -1) _h = total / _w;
				top_blob = bottom_blob.reshape(_w, _h, opt.blob_allocator);
			}
			else if (ndim == 3)
			{
				int _w = w;
				int _h = h;
				int _c = c;
				if (_w == 0) _w = bottom_blob.w;
				if (_h == 0) _h = bottom_blob.h;
				if (_c == 0) _c = bottom_blob.c;

				if (_w == -1) _w = total / _c / _h;
				if (_h == -1) _h = total / _c / _w;
				if (_c == -1) _c = total / _h / _w;
				top_blob = bottom_blob.reshape(_w, _h, _c, opt.blob_allocator);
			}

			if (top_blob.empty())
				return -100;

			return 0;
		}
	}
	namespace GPU {

	}
}