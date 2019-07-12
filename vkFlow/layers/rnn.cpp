#include "rnn.h"

namespace backend {
	namespace CPU {
		RNN::RNN() {
			one_blob_only = false;
			support_inplace = false;
		}


		int RNN::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
			const Mat& input_blob = bottom_blobs[0];
			size_t elemsize = input_blob.elemsize;
			const Mat& cont_blob = bottom_blobs[1];

			int T = input_blob.c;
			int size = input_blob.w;

			Mat hidden(num_output, 4u, opt.workspace_allocator);
			if (hidden.empty())
				return -100;
			hidden.fill(0.f);
			Mat& top_blob = top_blobs[0];
			top_blob.create(num_output, 1, T, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			// unroll
			for (int t = 0; t < T; t++)	{
				// clip hidden by continuation indicator
				// h_cont_{t-1} = cont_t * h_{t-1}
				// h_cont_{t-1} = h_{t-1} if cont_t == 1
				//                0       otherwise
				// calculate hidden
				// h_t = tanh( W_hh * h_cont_{t-1} + W_xh * x_t + b_h )
				const float cont = cont_blob[t];
				const Mat x = input_blob.channel(t);
				float* hidden_data = hidden;
				for (int q = 0; q < num_output; q++) {
					float h_cont = cont ? hidden_data[q] : 0.f;

					const float* weight_hh_data_ptr = (const float*)weight_hh_data + weight_hh_data.w * q;
					const float* weight_xh_data_ptr = (const float*)weight_xh_data + weight_xh_data.w * q;
					const float* x_data = x;

					float s0 = bias_h_data[q];
					for (int i = 0; i < size; i++)
					{
						s0 += weight_hh_data_ptr[i] * h_cont + weight_xh_data_ptr[i] * x_data[i];
					}

					hidden_data[q] = tanh(s0);
				}

				// o_t = tanh( W_ho * h_t + b_o )
				Mat output = top_blob.channel(t);
				float* output_data = output;
				for (int q = 0; q < num_output; q++) {
					const float* weight_ho_data_ptr = (const float*)weight_ho_data + weight_ho_data.w * q;
					float s0 = bias_o_data[q];
					for (int i = 0; i < size; i++)
						s0 += weight_ho_data_ptr[i] * hidden_data[i];
	
					output_data[q] = tanh(s0);
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}