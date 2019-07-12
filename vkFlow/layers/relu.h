#ifndef RELU_LAYER_H
#define RELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ReLU : public Layer {
		public:
			ReLU();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
			virtual int forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const;

			float slope;
		};
	}
	namespace GPU {
		class ReLU : virtual public CPU::ReLU {

		public:
			ReLU();

		};
	}
}

#endif

