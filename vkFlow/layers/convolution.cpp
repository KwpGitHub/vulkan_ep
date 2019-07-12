#include "convolution.h"
#include <algorithm>
#include "quantize.h"
#include "requantize.h"
#include "innerproduct.h"

namespace backend {
	namespace CPU {

		Convolution::Convolution() {
			one_blob_only = true;
			support_inplace = false;
			use_int8_requantize = false;
			quantize = 0;
		}
		
		int Convolution::create_pipeline(const Option& opt)	{
			Option opt_cpu = opt;
			opt_cpu.use_vulkan_compute = false;
			use_int8_inference = opt.use_int8_inference;

			if (int8_scale_term == 0)
				use_int8_inference = false;

			bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
			bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);
			if (weight_data_is_int8 && !use_int8_inference) return -1;
			
			return 0;
		}

		int Convolution::destroy_pipeline(const Option& opt)
		{
			Option opt_cpu = opt;
			opt_cpu.use_vulkan_compute = false;

			for (int i = 0; i < (int)dequantize_ops.size(); i++)
			{
				dequantize_ops[i]->destroy_pipeline(opt_cpu);
				delete dequantize_ops[i];
			}
			dequantize_ops.clear();

			for (int i = 0; i < (int)requantize_ops.size(); i++)
			{
				requantize_ops[i]->destroy_pipeline(opt_cpu);
				delete requantize_ops[i];
			}
			requantize_ops.clear();

			dequantize_scales.clear();
			requantize_scales.clear();

			return 0;
		}

		int Convolution::create_requantize_op(void)
		{
			/*
			if (!use_int8_requantize)
			{
				fprintf(stderr, "requantized op set but use_int8_requantize disabled\n");
				return -1;
			}

			requantize_ops.resize(num_output);
			for (int n = 0; n < num_output; n++)
			{
				requantize_ops[n] = ncnn::create_layer(ncnn::LayerType::Requantize);

				float scale_in = 1.f;
				float scale_out = 1.f;

				if (weight_data_int8_scales[n] == 0)
				{
					scale_in = 0;
				}
				else
				{
					scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[n]);
				}

				scale_out = top_blob_int8_scale;

				ncnn::ParamDict pd;
				pd.set(0, scale_in);   // scale in
				pd.set(1, scale_out);  // scale_out
				pd.set(2, bias_term);  // bias_term
				pd.set(3, 1);          // bias_data_size

				requantize_ops[n]->load_param(pd);

				ncnn::Mat weights[1];
				weights[0] = bias_data.range(n, 1);

				requantize_ops[n]->load_model(ModelBinFromMatArray(weights));

				requantize_scales.push_back(scale_in);
				requantize_scales.push_back(scale_out);
			}
			*/
			return 0;
		}

		int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
		{
			// convolv with NxN kernel
			// value = value + bias

			// flattened blob, implement as InnerProduct
			if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
			{
				int num_input = weight_data_size / num_output;
				if (bottom_blob.w == num_input)
				{
					// call InnerProduct
					backend::Layer* op = new CPU::InnerProduct();

					// set param
					backend::ParamDict pd;
					pd.set(0, num_output);
					pd.set(1, bias_term);
					pd.set(2, weight_data_size);
					pd.set(8, int8_scale_term);

					// set weights
					backend::Mat weights[4];
					weights[0] = weight_data;
					weights[1] = bias_data;

					if (int8_scale_term)
					{
						weights[2] = weight_data_int8_scales;
						weights[3] = Mat(1, (size_t)4u, (void*)& bottom_blob_int8_scale);
					}

					Option opt_cpu = opt;
					opt_cpu.use_vulkan_compute = false;
					op->create_pipeline(opt_cpu);
					op->forward(bottom_blob, top_blob, opt);

					delete op;

					return 0;
				}
			}

			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
			const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

			Mat bottom_blob_unbordered = bottom_blob;
			if (use_int8_inference && elemsize != 1)
			{
				Mat bottom_blob_int8;
				bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
				if (bottom_blob_int8.empty())
					return -100;

				// quantize, scale and round to nearest
				{
					backend::Option opt_g = opt;
					opt_g.blob_allocator = bottom_blob_int8.allocator;

					quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
				}

				bottom_blob_unbordered = bottom_blob_int8;
			}

			Mat bottom_blob_bordered = bottom_blob_unbordered;
			if (pad_w > 0 || pad_h > 0)
			{
				copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
				if (bottom_blob_bordered.empty())
					return -100;

				w = bottom_blob_bordered.w;
				h = bottom_blob_bordered.h;
			}
			else if (pad_w == -233 && pad_h == -233)
			{
				int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
				int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
				if (wpad > 0 || hpad > 0)
				{
					copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
					if (bottom_blob_bordered.empty())
						return -100;
				}

				w = bottom_blob_bordered.w;
				h = bottom_blob_bordered.h;
			}

			int outw = (w - kernel_extent_w) / stride_w + 1;
			int outh = (h - kernel_extent_h) / stride_h + 1;

			const int maxk = kernel_w * kernel_h;

			// kernel offsets
			std::vector<int> _space_ofs(maxk);
			int* space_ofs = &_space_ofs[0];
			{
				int p1 = 0;
				int p2 = 0;
				int gap = w * dilation_h - kernel_w * dilation_w;
				for (int i = 0; i < kernel_h; i++)
				{
					for (int j = 0; j < kernel_w; j++)
					{
						space_ofs[p1] = p2;
						p1++;
						p2 += dilation_w;
					}
					p2 += gap;
				}
			}

			// float32
			top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			// num_output
#pragma omp parallel for num_threads(opt.num_threads)
			for (int p = 0; p < num_output; p++)
			{
				float* outptr = top_blob.channel(p);

				for (int i = 0; i < outh; i++)
				{
					for (int j = 0; j < outw; j++)
					{
						float sum = 0.f;

						if (bias_term)
							sum = bias_data[p];

						const float* kptr = (const float*)weight_data + maxk * channels * p;

						// channels
						for (int q = 0; q < channels; q++)
						{
							const Mat m = bottom_blob_bordered.channel(q);
							const float* sptr = m.row(i * stride_h) + j * stride_w;

							for (int k = 0; k < maxk; k++) // 29.23
							{
								float val = sptr[space_ofs[k]]; // 20.72
								float w = kptr[k];
								sum += val * w; // 41.45
							}

							kptr += maxk;
						}

						if (activation_type == 1)
						{
							sum = std::max<float>(sum, 0.f);
						}
						else if (activation_type == 2)
						{
							float slope = activation_params[0];
							sum = sum > 0.f ? sum : sum * slope;
						}
						else if (activation_type == 3)
						{
							float min = activation_params[0];
							float max = activation_params[1];
							if (sum < min)
								sum = min;
							if (sum > max)
								sum = max;
						}
						else if (activation_type == 4)
						{
							sum = 1.f / (1.f + exp(-sum));
						}

						outptr[j] = sum;
					}

					outptr += outw;
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}