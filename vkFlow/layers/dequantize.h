#ifndef DEQUANTIZE_LAYER_H
#define DEQUANTIZE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Dequantize : public Layer {
		public:
			Dequantize();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float scale;
			int bias_term;
			int bias_data_size;

			Mat bias_data;
		};
	}
	namespace GPU {
		class Dequantize : virtual public CPU::Dequantize {

		public:
			Dequantize();
		};
	}
}

#endif

