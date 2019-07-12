#include "lstm.h"

namespace backend {
	namespace CPU {
		LSTM::LSTM() {
			one_blob_only = false;
			support_inplace = false;
		}


		int LSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
			const Mat& input_blob = bottom_blobs[0];
			size_t elemsize = input_blob.elemsize;
			const Mat& cont_blob = bottom_blobs[1];
			int T = input_blob.h;
			int size = input_blob.w;
			Mat hidden(num_output, 4u, opt.workspace_allocator);
			if (hidden.empty())
				return -100;
			hidden.fill(0.f);
			Mat cell(num_output, 4u, opt.workspace_allocator);
			if (cell.empty())
				return -100;
			Mat gates(4, num_output, 4u, opt.workspace_allocator);
			if (gates.empty())
				return -100;
			Mat& top_blob = top_blobs[0];
			top_blob.create(num_output, T, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;
			for (int t = 0; t < T; t++) {
				// h_cont_{t-1} = cont_t * h_{t-1}
				// h_cont_{t-1} = h_{t-1} if cont_t == 1
				//                0       otherwise
				// calculate hidden
				// gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
				const int cont = ((const int*)cont_blob)[t];
				const float* x = input_blob.row(t);
				for (int q = 0; q < num_output; q++) {
					//float h_cont = cont ? hidden[q] : 0.f;

					const float* I_bias_c_data_ptr = (const float*)bias_c_data;
					const float* F_bias_c_data_ptr = (const float*)bias_c_data + num_output;
					const float* O_bias_c_data_ptr = (const float*)bias_c_data + 2 * num_output;
					const float* G_bias_c_data_ptr = (const float*)bias_c_data + 3 * num_output;

					//const float* bias_c_data_ptr = (const float*)bias_c_data + 4 * q;
					float* gates_data = (float*)gates + 4 * q;
					const float* weight_hc_data_I = (const float*)weight_hc_data + weight_hc_data.w * q;
					const float* weight_xc_data_I = (const float*)weight_xc_data + weight_xc_data.w * q;
					const float* weight_hc_data_F = (const float*)weight_hc_data + weight_hc_data.w * q + num_output * num_output;
					const float* weight_xc_data_F = (const float*)weight_xc_data + weight_xc_data.w * q + num_output * size;
					const float* weight_hc_data_O = (const float*)weight_hc_data + weight_hc_data.w * q + num_output * num_output * 2;
					const float* weight_xc_data_O = (const float*)weight_xc_data + weight_xc_data.w * q + num_output * size * 2;
					const float* weight_hc_data_G = (const float*)weight_hc_data + weight_hc_data.w * q + num_output * num_output * 3;
					const float* weight_xc_data_G = (const float*)weight_xc_data + weight_xc_data.w * q + num_output * size * 3;

					float I = I_bias_c_data_ptr[q];
					float F = F_bias_c_data_ptr[q];
					float O = O_bias_c_data_ptr[q];
					float G = G_bias_c_data_ptr[q];

					for (int i = 0; i < size; i++) {
						I += weight_xc_data_I[i] * x[i];
						F += weight_xc_data_F[i] * x[i];
						O += weight_xc_data_O[i] * x[i];
						G += weight_xc_data_G[i] * x[i];
					}

					for (int i = 0; i < num_output; ++i) {
						I += weight_hc_data_I[i] * (cont == 0 ? 0 : hidden[i]);
						F += weight_hc_data_F[i] * (cont == 0 ? 0 : hidden[i]);
						O += weight_hc_data_O[i] * (cont == 0 ? 0 : hidden[i]);
						G += weight_hc_data_G[i] * (cont == 0 ? 0 : hidden[i]);
					}

					gates_data[0] = I;
					gates_data[1] = F;
					gates_data[2] = O;
					gates_data[3] = G;
				}

				float* output_data = top_blob.row(t);
				for (int q = 0; q < num_output; q++) {
					float* gates_data = (float*)gates + 4 * q;
					float I = gates_data[0];
					float F = gates_data[1];
					float O = gates_data[2];
					float G = gates_data[3];
					I = 1.f / (1.f + exp(-I));
					F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
					O = 1.f / (1.f + exp(-O));
					G = tanh(G);
					float cell2 = cont ? F * cell[q] + I * G : I * G;
					float H = O * tanh(cell2);
					cell[q] = cell2;
					hidden[q] = H;
					output_data[q] = H;
				}
			}
			return 0;

	}
	namespace GPU {

	}
}