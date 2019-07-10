#include "mat.h"
namespace backend {

	void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
	{
		backend::Layer* op;

		if (mean_vals && !norm_vals)
		{
			// substract mean only
			op = backend::Bias;

			backend::ParamDict pd;
			pd.set(0, c);

			op->load_param(pd);

			backend::Mat weights[1];
			weights[0] = Mat(c);
			for (int q = 0; q < c; q++)
			{
				weights[0][q] = -mean_vals[q];
			}

			//op->load_model(backend::ModelBinFromMatArray(weights));
		}
		else if (!mean_vals && norm_vals)
		{
			// normalize only
			op = backend::Scale;

			backend::ParamDict pd;
			pd.set(0, c);

			op->load_param(pd);

			backend::Mat weights[1];
			weights[0] = Mat(c);
			for (int q = 0; q < c; q++)
			{
				weights[0][q] = norm_vals[q];
			}

			//op->load_model(ncnn::ModelBinFromMatArray(weights));
		}
		else if (mean_vals && norm_vals)
		{
			// substract mean and normalize
			op = backend::Scale;

			backend::ParamDict pd;
			pd.set(0, c);
			pd.set(1, 1);

			op->load_param(pd);

			backend::Mat weights[2];
			weights[0] = Mat(c);
			weights[1] = Mat(c);
			for (int q = 0; q < c; q++)
			{
				weights[0][q] = norm_vals[q];
				weights[1][q] = -mean_vals[q] * norm_vals[q];
			}

			//op->load_model(ncnn::ModelBinFromMatArray(weights));
		}
		else // if (!mean_vals && !norm_vals)
		{
			return;
		}

		op->forward_inplace(*this);

		delete op;
	}

	static float half2float(unsigned short value)
	{	unsigned short sign = (value & 0x8000) >> 15;
		unsigned short exponent = (value & 0x7c00) >> 10;
		unsigned short significand = value & 0x03FF;

		// 1 : 8 : 23
		union
		{
			unsigned int u;
			float f;
		} tmp;
		if (exponent == 0)
		{
			if (significand == 0)
			{
				tmp.u = (sign << 31);
			}
			else
			{
				exponent = 0;
				while ((significand & 0x200) == 0)
				{
					significand <<= 1;
					exponent++;
				}
				significand <<= 1;
				significand &= 0x3FF;
				tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
			}
		}
		else if (exponent == 0x1F)
		{
			tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
		}
		else
		{
			tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
		}

		return tmp.f;
	}
	Mat Mat::from_float16(const unsigned short* data, int size)
	{
		Mat m(size);
		if (m.empty())
			return m;

		float* ptr = m;
		int remain = size;

		for (; remain > 0; remain--)
		{
			*ptr = half2float(*data);
			data++;
			ptr++;
		}

		return m;
	}

	void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, Allocator* allocator, int num_threads)
	{
		backend::Layer* padding = backend::Padding;

		backend::ParamDict pd;
		pd.set(0, top);
		pd.set(1, bottom);
		pd.set(2, left);
		pd.set(3, right);
		pd.set(4, type);
		pd.set(5, v);

		padding->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		padding->forward(src, dst, opt);

		delete padding;
	}

	void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, Allocator* allocator, int num_threads)
	{
		backend::Layer* crop = backend::Crop;

		backend::ParamDict pd;
		pd.set(0, left);
		pd.set(1, top);
		pd.set(2, 0);
		pd.set(3, src.w - left - right);
		pd.set(4, src.h - top - bottom);
		pd.set(5, src.c);

		crop->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		crop->forward(src, dst, opt);

		delete crop;
	}

	void resize_bilinear(const Mat& src, Mat& dst, int w, int h, Allocator* allocator, int num_threads)
	{
		backend::Layer* interp = backend::Interp);

		backend::ParamDict pd;
		pd.set(0, 2);
		pd.set(3, h);
		pd.set(4, w);

		interp->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		interp->forward(src, dst, opt);

		delete interp;
	}

	void resize_bicubic(const Mat& src, Mat& dst, int w, int h, Allocator* allocator, int num_threads)
	{
		backend::Layer* interp = backend::Interps;

		backend::ParamDict pd;
		pd.set(0, 3);
		pd.set(3, h);
		pd.set(4, w);

		interp->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		interp->forward(src, dst, opt);

		delete interp;
	}

	void convert_packing(const Mat& src, Mat& dst, int _packing, Allocator* allocator, int num_threads)
	{
		backend::Layer* packing = backend::Packing);

		backend::ParamDict pd;
		pd.set(0, _packing);

		packing->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		packing->forward(src, dst, opt);

		delete packing;
	}

	void cast_float32_to_float16(const Mat& src, Mat& dst, Allocator* allocator, int num_threads)
	{
		backend::Layer* cast = backend::Cast);

		backend::ParamDict pd;
		pd.set(0, 1);
		pd.set(1, 2);

		cast->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		cast->forward(src, dst, opt);

		delete cast;
	}

	void cast_float16_to_float32(const Mat& src, Mat& dst, Allocator* allocator, int num_threads)
	{
		backend::Layer* cast = backend::Cast);

		backend::ParamDict pd;
		pd.set(0, 2);
		pd.set(1, 1);

		cast->load_param(pd);

		backend::Option opt;
		opt.num_threads = num_threads;
		opt.blob_allocator = allocator;

		cast->forward(src, dst, opt);

		delete cast;
	}



}