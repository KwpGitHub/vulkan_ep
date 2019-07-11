#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class SoftMax : public Layer {
		public:
			SoftMax();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int axis;
		};
	}
	namespace GPU {
		class SoftMax : virtual public CPU::SoftMax {

		public:
			SoftMax();

		};
	}
}

#endif

