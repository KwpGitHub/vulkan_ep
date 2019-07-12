#include "quantize.h"

namespace backend {
	namespace CPU {
		Quantize::Quantize() {
			one_blob_only = true;
			support_inplace = false;
		}

		static inline signed char float2int8(float v) {
			int int32 = round(v);
			if (int32 > 127) return 127;
			if (int32 < -128) return -128;
			return (signed char)int32;
		}

		int Quantize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int dims = bottom_blob.dims;
			if (dims == 1) {
				int w = bottom_blob.w;
				top_blob.create(w, (size_t)1u, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				const float* ptr = bottom_blob;
				signed char* outptr = top_blob;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int i = 0; i < w; i++)
					outptr[i] = float2int8(ptr[i] * scale);
			}

			if (dims == 2) {
				int w = bottom_blob.w;
				int h = bottom_blob.h;
				int size = w * h;
				top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				const float* ptr = bottom_blob;
				signed char* outptr = top_blob;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int i = 0; i < size; i++)
					outptr[i] = float2int8(ptr[i] * scale);
			}

			if (dims == 3) {
				int w = bottom_blob.w;
				int h = bottom_blob.h;
				int channels = bottom_blob.c;
				int size = w * h;

				top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					signed char* outptr = top_blob.channel(q);
					for (int i = 0; i < size; i++) {
						outptr[i] = float2int8(ptr[i] * scale);
					}
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}