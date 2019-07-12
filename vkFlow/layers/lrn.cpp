#include "lrn.h"

namespace backend {
	namespace CPU {
		Lrn::Lrn() {
			one_blob_only = true;
			support_inplace = true;
		}


		int Lrn::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			size_t elemsize = bottom_top_blob.elemsize;
			int size = w * h;
			Mat square_blob;
			square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
			if (square_blob.empty())
				return -100;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				const float* ptr = bottom_top_blob.channel(q);
				float* outptr = square_blob.channel(q);
				for (int i = 0; i < size; i++) {
					outptr[i] = ptr[i] * ptr[i];
				}
			}

			if (region_type == NormRegion_ACROSS_CHANNELS)
			{
				Mat square_sum;
				square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
				if (square_sum.empty())
					return -100;
				square_sum.fill(0.f);
				const float alpha_div_size = alpha / local_size;
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ssptr = square_sum.channel(q);
					for (int p = q - local_size / 2; p <= q + local_size / 2; p++) {
						if (p < 0 || p >= channels)
							continue;
						const float* sptr = square_blob.channel(p);
						for (int i = 0; i < size; i++)
							ssptr[i] += sptr[i];
					}
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++)
						ptr[i] = ptr[i] * pow(bias + alpha_div_size * ssptr[i], -beta);
				}
			}
			else if (region_type == NormRegion_WITHIN_CHANNEL) {
				int outw = w;
				int outh = h;
				Mat square_blob_bordered = square_blob;
				int pad = local_size / 2;
				if (pad > 0)
				{
					copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
					if (square_blob_bordered.empty())
						return -100;

					w = square_blob_bordered.w;
					h = square_blob_bordered.h;
				}

				const int maxk = local_size * local_size;
				const float alpha_div_size = alpha / maxk;
				std::vector<int> _space_ofs(maxk);
				int* space_ofs = &_space_ofs[0];
				{
					int p1 = 0;
					int p2 = 0;
					int gap = w - local_size;
					for (int i = 0; i < local_size; i++) {
						for (int j = 0; j < local_size; j++) {
							space_ofs[p1] = p2;
							p1++;
							p2++;
						}
						p2 += gap;
					}
				}

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					const Mat m = square_blob_bordered.channel(q);
					for (int i = 0; i < outh; i++) {
						for (int j = 0; j < outw; j++) {
							const float* sptr = m.row(i) + j;
							float ss = 0.f;
							for (int k = 0; k < maxk; k++) {
								float val = sptr[space_ofs[k]];
								ss += val;
							}
							ptr[j] = ptr[j] * pow(bias + alpha_div_size * ss, -beta);
						}
						ptr += outw;
					}
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}