#ifndef INNERPRODUCT_LAYER_H
#define INNERPRODUCT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class InnerProduct : public Layer {
		public:
			InnerProduct();
			virtual int create_pipeline(const Option& opt);
			virtual int destroy_pipeline(const Option& opt);
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int num_ouput;
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
			bool use_int8_inference;
			backend::Layer* quantize;
			std::vector<backend::Layer*> dequantize_ops;
		};
	}
	namespace GPU {
		class InnerProduct : virtual public CPU::InnerProduct {

		public:
			InnerProduct();
		};
	}
}

#endif

