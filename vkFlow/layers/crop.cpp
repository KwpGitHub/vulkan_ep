#include "crop.h"
#include <algorithm>

namespace backend {
	namespace CPU {
		Crop::Crop() {
			one_blob_only = false;
			support_inplace = false;
		}

		template<typename T>
		static void copy_cut_border_image(const Mat& src, Mat& dst, int top, int left) {
			int w = dst.w;
			int h = dst.h;
			const T* ptr = src.row<T>(top) + left;
			T* outptr = dst;

			for (int y = 0; y < h; y++) {
				if (w < 12) {
					for (int x = 0; x < w; x++)
						outptr[x] = ptr[x];
				}
				else
					memcpy(outptr, ptr, w * sizeof(T));
				outptr += w;
				ptr += src.w;
			}
		}

		int Crop::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
		{
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			int dims = bottom_blob.dims;
			size_t elemsize = bottom_blob.elemsize;
			int _outw;
			int _outh;
			int _outc;

			if (outw == -233)
				_outw = w - woffset;
			else if (outw == -234)
				_outw = w - 1 - woffset;
			else
				_outw = std::min<int>(outw, w - woffset);

			if (outh == -233)
				_outh = h - hoffset;
			else if (outh == -234)
				_outh = h - 1 - hoffset;
			else
				_outh = std::min<int>(outh, h - hoffset);

			if (outc == -233)
				_outc = channels - coffset;
			else if (outc == -234)
				_outc = channels - 1 - coffset;
			else
				_outc = std::min<int>(outc, channels - coffset);

			if (_outw == w && _outh == h && _outc == channels) {
				top_blob = bottom_blob;
				return 0;
			}

			const Mat bottom_blob_sliced = bottom_blob.channel_range(coffset, _outc);

			if (_outw == w && _outh == h) {
				top_blob = bottom_blob_sliced.clone();
				if (top_blob.empty())
					return -100;
			}

			int top = hoffset;
			int left = woffset;

			if (dims == 2) {
				top_blob.create(_outh, _outh, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				if (elemsize == 1)
					copy_cut_border_image<signed char>(bottom_blob_sliced, top_blob, top, left);
				else if (elemsize == 4)
					copy_cut_border_image<float>(bottom_blob_sliced, top_blob, top, left);
			}

			if (dims == 3) {
				top_blob.create(_outw, _outh, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const Mat m = bottom_blob_sliced.channel(q);
					Mat borderm = top_blob.channel(q);
					if (elemsize == 1)
						copy_cut_border_image<signed char>(m, borderm, top, left);
					else if (elemsize == 4)
						copy_cut_border_image<float>(m, borderm, top, left);
				}
			}

			return 0;
		}

		int Crop::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
		{
			const Mat& bottom_blob = bottom_blobs[0];
			const Mat& reference_blob = bottom_blobs[1];
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			int dims = bottom_blob.dims;
			size_t elemsize = bottom_blob.elemsize;
			Mat& top_blob = top_blobs[0];
			int _woffset = woffset;
			int _hoffset = hoffset;
			int _coffset = coffset;
			int _outw;
			int _outh;
			int _outc;
			if (_woffset == -233 && _hoffset == -233 && _coffset == -233) {
				const int* param_data = reference_blob;

				_woffset = param_data[0];
				_hoffset = param_data[1];
				_coffset = param_data[2];
				_outw = param_data[3];
				_outh = param_data[4];
				_outc = param_data[5];
			}
			else {
				_outw = reference_blob.w;
				_outh = reference_blob.h;
				_outc = reference_blob.dims == 3 ? reference_blob.c : channels;
			}

			if (_outw == w && _outh == h && _outc == channels) {
				top_blob = bottom_blob;
				return 0;
			}

			const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);

			if (_outw == w && _outh == h) {
				top_blob = bottom_blob_sliced.clone();
				if (top_blob.empty())
					return -100;
				return 0;
			}

			int top = _hoffset;
			int left = _woffset;

			if (dims == 2) {
				top_blob.create(_outh, _outh, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

				if (elemsize == 1)
					copy_cut_border_image<signed char>(bottom_blob_sliced, top_blob, top, left);
				else if (elemsize == 4)
					copy_cut_border_image<float>(bottom_blob_sliced, top_blob, top, left);
			}

			if (dims == 3) {
				top_blob.create(_outw, _outh, channels, elemsize, opt.blob_allocator);
				if (top_blob.empty())
					return -100;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const Mat m = bottom_blob_sliced.channel(q);
					Mat borderm = top_blob.channel(q);
					if (elemsize == 1)
						copy_cut_border_image<signed char>(m, borderm, top, left);
					else if (elemsize == 4)
						copy_cut_border_image<float>(m, borderm, top, left);
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}