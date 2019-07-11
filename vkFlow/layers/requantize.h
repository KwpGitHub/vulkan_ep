#ifndef REQUANTIZE_LAYER_H
#define REQUANTIZE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Requantize : public Layer {
		public:
			Requantize();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			float scale_in;	// bottom_blob_scale * weight_scale
			float scale_out;// top_blob_scale / (bottom_blob_scale * weight_scale)
			int bias_term;
			int bias_data_size;
			bool fusion_relu;

			Mat bias_data;
		};
	}
	namespace GPU {
		class Requantize : virtual public CPU::Requantize {

		public:
			Requantize();

		};
	}
}

#endif

