#include "cast.h"

namespace backend {
	namespace CPU {
		Cast::Cast() {
			one_blob_only = true;
			support_inplace = false;
		}

		static unsigned short float32_to_float16(float value) {
			union {
				unsigned int u;
				float f;
			} tmp;

			tmp.f = value;
			unsigned short sign = (tmp.u & 0x80000000) >> 31;
			unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
			unsigned int significand = tmp.u & 0x7FFFFF;
			unsigned short fp16;

			if (exponent == 0)
				fp16 = (sign << 15) | (0x00 << 10) | 0x00;
			else if (exponent == 0xFF)
				fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
			else {
				short newexp = exponent + (-127 + 15);
				if (newexp >= 31)
					fp16 = (sign << 15) | (0x1F << 10) | 0x00;
				else if (newexp <= 0) {
					if (newexp >= -10) {
						unsigned short sig = (significand | 0x800000) >> (14 - newexp);
						fp16 = (sign << 15) | (0x00 << 10) | sig;
					}
					else
						fp16 = (sign << 15) | (0x00 << 10) | 0x00;
				}
				else
					fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
			}

			return fp16;
		}
		
		static float float16_to_float32(unsigned short value) {
			unsigned short sign = (value & 0x8000) >> 15;
			unsigned short exponent = (value & 0x7c00) >> 10;
			unsigned short significand = value & 0x03FF;
			union {
				unsigned int u;
				float f;
			} tmp;

			if (exponent == 0)
			{
				if (significand == 0)
					tmp.u = (sign << 31);
				else {
					exponent = 0;
					while ((significand & 0x200) == 0) {
						significand <<= 1;
						exponent++;
					}
					significand <<= 1;
					significand &= 0x3FF;
					tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
				}
			}
			else if (exponent == 0x1F)		
				tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
			else
				tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
		
			return tmp.f;
		}

		static signed char float32_to_int8(float value)	{
			float tmp;
			if (value >= 0.f) tmp = value + 0.5;
			else tmp = value - 0.5;
			if (tmp > 127) return 127;
			if (tmp < -128)	return -128;
			return tmp;
		}

		int Cast::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			if (type_from == type_to) {
				top_blob = bottom_blob;
				return 0;
			}

			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			int dims = bottom_blob.dims;
			size_t elemsize = bottom_blob.elemsize;
			int packing = bottom_blob.packing;

			size_t out_elemsize = elemsize;
			if (type_to == 1)
				out_elemsize = 4 * packing;
			else if (type_to == 2)
				out_elemsize = 2 * packing;
			else if (type_to == 3)
				out_elemsize = packing;
	
			if (dims == 1)
				top_blob.create(w, out_elemsize, packing, opt.blob_allocator);
			else if (dims == 2)
				top_blob.create(w, h, out_elemsize, packing, opt.blob_allocator);
			else if (dims == 3)
				top_blob.create(w, h, channels, out_elemsize, packing, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			int size = w * h * packing;

			if (type_from == 1 && type_to == 2) {
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_blob.channel(q);
					unsigned short* outptr = top_blob.channel(q);
					for (int i = 0; i < size; i++)
						outptr[i] = float32_to_float16(ptr[i]);
				}
			}

			if (type_from == 2 && type_to == 1)	{
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const unsigned short* ptr = bottom_blob.channel(q);
					float* outptr = top_blob.channel(q);
					for (int i = 0; i < size; i++)
						outptr[i] = float16_to_float32(ptr[i]);
					
				}
			}

			// TODO more cast type

			return 0;
		}
	}
	namespace GPU {

	}
}