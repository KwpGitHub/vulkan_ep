#ifndef CONVOLUTIONDEPTHWISE_LAYER_H
#define CONVOLUTIONDEPTHWISE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ConvolutionDepthwise : public Layer {
		public:
			ConvolutionDepthwise();
			virtual int create_pipeline(const Option& opt);
			virtual int destroy_pipeline(const Option& opt);
			virtual int create_requantize_op(void);
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int num_output; int kernel_w, kernel_h; int dilation_w, dilation_h; int stride_w, stride_h; int pad_w, pad_h;
			int bias_term;
			int weight_data_size;
			int int8_scale_term;
			// 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
			int activation_type;

			Mat activation_params;
			Mat weight_data;
			Mat bias_data;
			Mat weight_data_int8_scales;
			float bottom_blob_int8_scale;
			float top_blob_int8_scale;
			bool use_int8_inference;
			bool use_int8_requantize;

			backend::Layer quantize;
			std::vector<backend::Layer*> dequantize_ops;
			std::vector<backend::Layer*> requantize_ops;
			std::vector<float> dequantize_scales;
			std::vector<float> requantize_scales;
		};
	}
	namespace GPU {
		class ConvolutionDepthwise : virtual public CPU::ConvolutionDepthwise {

		public:
			ConvolutionDepthwise();

		};
	}
}

#endif

