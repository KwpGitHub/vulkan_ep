#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Softmax : public Layer {
		public:
			Softmax();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int axis;
		};
	}
	namespace GPU {
		class Softmax : virtual public CPU::Softmax {

		public:
			Softmax();

		};
	}
}

#endif

