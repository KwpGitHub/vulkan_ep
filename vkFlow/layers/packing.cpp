#include "packing.h"

namespace backend {
	namespace CPU {
		Packing::Packing() {
			one_blob_only = true;
			support_inplace = false;
		}

		int Packing::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int packing = bottom_blob.packing;
			if (packing == out_packing) {
				top_blob = bottom_blob;
				return 0;
			}

			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			int dims = bottom_blob.dims;
			size_t elemsize = bottom_blob.elemsize;

			if (!use_padding) {
				if (dims == 1 && w * packing % out_packing != 0) {
					top_blob = bottom_blob;
					return 0;
				}
				if (dims == 2 && h * packing % out_packing != 0) {
					top_blob = bottom_blob;
					return 0;
				}
				if (dims == 3 && channels * packing % out_packing != 0) {
					top_blob = bottom_blob;
					return 0;
				}
			}

			if (dims == 1) {
				if (out_packing == 1) {
					top_blob = bottom_blob;
					top_blob.w = w * packing;
					top_blob.cstep = w * packing;
					top_blob.elemsize = elemsize / packing;
					top_blob.packing = out_packing;
					return 0;
				}

				int outw = (w * packing + out_packing - 1) / out_packing;
				size_t out_elemsize = elemsize / packing * out_packing;

				top_blob.create(outw, out_elemsize, out_packing, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				memcpy(top_blob.data, bottom_blob.data, w * elemsize);

				return 0;
			}

			if (dims == 2) {
				int outh = (h * packing + out_packing - 1) / out_packing;
				size_t out_elemsize = elemsize / packing * out_packing;
				size_t lane_size = out_elemsize / out_packing;

				top_blob.create(w, outh, out_elemsize, out_packing, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for
				for (int i = 0; i < outh; i++) {
					unsigned char* outptr = (unsigned char*)top_blob + i * w * out_elemsize;

					for (int j = 0; j < w; j++)	{
						unsigned char* out_elem_ptr = outptr + j * out_elemsize;
						for (int k = 0; k < out_packing; k++) {
							int srcy = (i * out_packing + k) / packing;
							if (srcy >= h)
								break;

							int srck = (i * out_packing + k) % packing;
							const unsigned char* ptr = (const unsigned char*)bottom_blob + srcy * w * elemsize;
							const unsigned char* elem_ptr = ptr + j * elemsize;
							memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
						}
					}
				}

				return 0;
			}

			if (dims == 3) {
				int outc = (channels * packing + out_packing - 1) / out_packing;
				size_t out_elemsize = elemsize / packing * out_packing;
				size_t lane_size = out_elemsize / out_packing;
				top_blob.create(w, h, outc, out_elemsize, out_packing, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for
				for (int q = 0; q < outc; q++) {
					Mat out = top_blob.channel(q);
					for (int i = 0; i < h; i++) {
						unsigned char* outptr = (unsigned char*)out + i * w * out_elemsize;
						for (int j = 0; j < w; j++) {
							unsigned char* out_elem_ptr = outptr + j * out_elemsize;
							for (int k = 0; k < out_packing; k++) {
								int srcq = (q * out_packing + k) / packing;
								if (srcq >= channels)
									break;

								int srck = (q * out_packing + k) % packing;
								const Mat m = bottom_blob.channel(srcq);
								const unsigned char* ptr = (const unsigned char*)m + i * w * elemsize;
								const unsigned char* elem_ptr = ptr + j * elemsize;
								memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
							}
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