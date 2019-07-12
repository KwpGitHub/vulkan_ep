#ifndef SELU_LAYER_H
#define SELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class SeLU : public Layer {
		public:
			SeLU();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float alpha, lambda;
		};
	}
	namespace GPU {
		class SeLU : virtual public CPU::SeLU {

		public:
			SeLU();


		};
	}
}

#endif

