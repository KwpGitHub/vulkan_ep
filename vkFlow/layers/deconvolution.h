#ifndef DECONVOLUTION_LAYER_H
#define DECONVOLUTION_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		/*class Deconvolution : public Deconvolution {
		public:
			Deconvolution();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int num_output;	int kernel_w, kernel_h; int dilation_w, dilation_h; int stride_w, stride_h; int pad_w, pad_h;
			int bias_term;
			int weight_data_size;
			// 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
			int activation_type;

			Mat activation_params;
			Mat weight_data;
			Mat bias_data;
		};*/
	}
	namespace GPU {
		//class Deconvolution : virtual public CPU::Deconvolution {

	//	public:
//			Deconvolution();

	//	};
	}
}

#endif

