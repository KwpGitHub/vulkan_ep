#include "embed.h"

namespace backend {
	namespace CPU {
		Embed::Embed() {
			one_blob_only = true;
			support_inplace = false;
		}

		int Embed::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int words = bottom_blob.total();
			top_blob.create(num_output, words, 4u, opt.blob_allocator);
			if (top_blob.empty())
				return -100;
#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < words; q++) {
				float* outptr = top_blob.row(q);
				int word_index = ((const int*)bottom_blob)[q];
				if (word_index < 0) word_index = 0;
				if (word_index >= input_dim) word_index = input_dim - 1;

				const float* em = (const float*)weight_data + num_output * word_index;
				memcpy(outptr, em, num_output * sizeof(float));

				if (bias_term) {
					for (int p = 0; p < num_output; p++) {
						outptr[p] += bias_data[p];
					}
				}
			}

			return 0;

		}

	}
	namespace GPU {

	}
}