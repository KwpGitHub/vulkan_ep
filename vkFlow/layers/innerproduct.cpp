#include "innerproduct.h"
#include <algorithm>

namespace backend {
	namespace CPU {
		InnerProduct::InnerProduct() {
			one_blob_only = true;
			support_inplace = false;
			quantize = 0;
		}
	
		int InnerProduct::create_pipeline(const Option& opt)
		{
			Option opt_cpu = opt;
			opt_cpu.use_vulkan_compute = false;
			use_int8_inference = opt.use_int8_inference;
			if (int8_scale_term == 0)
				use_int8_inference = false;

			bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
			bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

			if (weight_data_is_int8 && !use_int8_inference){
				fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
				return -1;
			}

			// runtime quantize the weight data
			return 0;
		}

		int InnerProduct::destroy_pipeline(const Option& opt)
		{
			Option opt_cpu = opt;
			opt_cpu.use_vulkan_compute = false;

			if (quantize)
			{
				quantize->destroy_pipeline(opt_cpu);
				delete quantize;
				quantize = 0;
			}

			for (int i = 0; i < (int)dequantize_ops.size(); i++)
			{
				dequantize_ops[i]->destroy_pipeline(opt_cpu);
				delete dequantize_ops[i];
			}
			dequantize_ops.clear();

			return 0;
		}

		int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
		{
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int channels = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int size = w * h;

			top_blob.create(num_output, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			if (use_int8_inference)
			{
				Mat bottom_blob_tm = bottom_blob;
				if (elemsize != 1)
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

					bottom_blob_tm = bottom_blob_int8;
				}

				// num_output
#pragma omp parallel for num_threads(opt.num_threads)
				for (int p = 0; p < num_output; p++)
				{
					int sum = 0;
					int* out = top_blob;

					// channels
					for (int q = 0; q < channels; q++)
					{
						const signed char* w = (const signed char*)weight_data + size * channels * p + size * q;
						const signed char* m = bottom_blob_tm.channel(q);

						for (int i = 0; i < size; i++)
						{
							sum += m[i] * w[i];
						}
					}

					out[p] = sum;
				}

#pragma omp parallel for num_threads(opt.num_threads)
				for (int p = 0; p < num_output; p++)
				{
					int* out_s32 = top_blob;
					float* out_f32 = top_blob;
					float top_rescale = 1.f;
					if (weight_data_int8_scales[p] == 0)
						top_rescale = 0;
					else
						top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

					if (bias_term)
						out_f32[p] = out_s32[p] * top_rescale + bias_data[p];
					else
						out_f32[p] = out_s32[p] * top_rescale;

					if (activation_type == 1)
					{
						out_f32[p] = std::max<float>(out_f32[p], 0.f);
					}
				}

				return 0;
			}

			// num_output
#pragma omp parallel for num_threads(opt.num_threads)
			for (int p = 0; p < num_output; p++)
			{
				float sum = 0.f;

				if (bias_term)
					sum = bias_data[p];

				// channels
				for (int q = 0; q < channels; q++)
				{
					const float* w = (const float*)weight_data + size * channels * p + size * q;
					const float* m = bottom_blob.channel(q);

					for (int i = 0; i < size; i++)
					{
						sum += m[i] * w[i];
					}
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

				top_blob[p] = sum;
			}

			return 0;
		}
	}
	namespace GPU {

	}
}