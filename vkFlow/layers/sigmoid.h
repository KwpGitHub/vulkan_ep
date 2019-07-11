#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Sigmoid : public Layer {
		public:
			Sigmoid();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class Sigmoid : virtual public CPU::Sigmoid {

		public:
			Sigmoid();



		};
	}
}

#endif

