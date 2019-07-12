#include "softmax.h"
#include <float.h>
#include <math.h>
#include <algorithm>

namespace backend {
	namespace CPU {

		Softmax::Softmax() {
			one_blob_only = true;
			support_inplace = true;
		}

		int Softmax::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int dims = bottom_top_blob.dims;
			size_t elemsize = bottom_top_blob.elemsize;

			if (dims == 1) {
				int w = bottom_top_blob.w;

				float* ptr = bottom_top_blob;

				float max = -FLT_MAX;
				for (int i = 0; i < w; i++)
					max = std::max<float>(max, ptr[i]);

				float sum = 0.f;
				for (int i = 0; i < w; i++) {
					ptr[i] = exp(ptr[i] - max);
					sum += ptr[i];
				}

				for (int i = 0; i < w; i++)
					ptr[i] /= sum;

				return 0;
			}

			if (dims == 2 && axis == 0) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				Mat max;
				max.create(w, elemsize, opt.workspace_allocator);
				if (max.empty())
					return -100;
				max.fill(-FLT_MAX);

				for (int i = 0; i < h; i++) {
					const float* ptr = bottom_top_blob.row(i);
					for (int j = 0; j < w; j++)
						max[j] = std::max<float>(max[j], ptr[j]);
				}

				Mat sum;
				sum.create(w, elemsize, opt.workspace_allocator);
				if (sum.empty())
					return -100;
				sum.fill(0.f);

				for (int i = 0; i < h; i++) {
					float* ptr = bottom_top_blob.row(i);
					for (int j = 0; j < w; j++) {
						ptr[j] = exp(ptr[j] - max[j]);
						sum[j] += ptr[j];
					}
				}

				for (int i = 0; i < h; i++) {
					float* ptr = bottom_top_blob.row(i);
					for (int j = 0; j < w; j++)	{
						ptr[j] /= sum[j];
					}
				}

				return 0;
			}

			if (dims == 2 && axis == 1)
			{
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;

				for (int i = 0; i < h; i++) {
					float* ptr = bottom_top_blob.row(i);
					float m = -FLT_MAX;
					for (int j = 0; j < w; j++)
						m = std::max<float>(m, ptr[j]);

					float s = 0.f;
					for (int j = 0; j < w; j++){
						ptr[j] = exp(ptr[j] - m);
						s += ptr[j];
					}
					for (int j = 0; j < w; j++)
						ptr[j] /= s;
				}

				return 0;
			}

			if (dims == 3 && axis == 0){		
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				int channels = bottom_top_blob.c;
				int size = w * h;

				Mat max;
				max.create(w, h, elemsize, opt.workspace_allocator);
				if (max.empty())
					return -100;
				max.fill(-FLT_MAX);
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_top_blob.channel(q);

					for (int i = 0; i < size; i++)
						max[i] = std::max<float>(max[i], ptr[i]);
				}

				Mat sum;
				sum.create(w, h, elemsize, opt.workspace_allocator);
				if (sum.empty())
					return -100;
				sum.fill(0.f);
				for (int q = 0; q < channels; q++)	{
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++) {
						ptr[i] = exp(ptr[i] - max[i]);
						sum[i] += ptr[i];
					}
				}

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					for (int i = 0; i < size; i++) 
						ptr[i] /= sum[i];
				}

				return 0;
			}

			if (dims == 3 && axis == 1)	{
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				int channels = bottom_top_blob.c;

				Mat max;
				max.create(w, channels, elemsize, opt.workspace_allocator);
				if (max.empty())
					return -100;
				max.fill(-FLT_MAX);
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					const float* ptr = bottom_top_blob.channel(q);
					float* maxptr = max.row(q);
					for (int i = 0; i < h; i++)	{
						for (int j = 0; j < w; j++) 
							maxptr[j] = std::max<float>(maxptr[j], ptr[j]);
						ptr += w;
					}
				}

				Mat sum;
				sum.create(w, channels, elemsize, opt.workspace_allocator);
				if (sum.empty())
					return -100;
				sum.fill(0.f);
#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);
					float* maxptr = max.row(q);
					float* sumptr = sum.row(q);

					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							ptr[j] = exp(ptr[j] - maxptr[j]);
							sumptr[j] += ptr[j];
						}
						ptr += w;
					}
				}

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++)	{
					float* ptr = bottom_top_blob.channel(q);
					float* sumptr = sum.row(q);

					for (int i = 0; i < h; i++)	{
						for (int j = 0; j < w; j++)
							ptr[j] /= sumptr[j];

						ptr += w;
					}
				}

				return 0;
			}

			if (dims == 3 && axis == 2) {
				int w = bottom_top_blob.w;
				int h = bottom_top_blob.h;
				int channels = bottom_top_blob.c;

#pragma omp parallel for num_threads(opt.num_threads)
				for (int q = 0; q < channels; q++) {
					float* ptr = bottom_top_blob.channel(q);

					for (int i = 0; i < h; i++) {
						float max = -FLT_MAX;
						for (int j = 0; j < w; j++)
							max = std::max<float>(max, ptr[j]);
	
						float sum = 0.f;
						for (int j = 0; j < w; j++)	{
							ptr[j] = exp(ptr[j] - max);
							sum += ptr[j];
						}

						for (int j = 0; j < w; j++)						
							ptr[j] /= sum;

						ptr += w;
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