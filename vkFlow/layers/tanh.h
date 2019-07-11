#ifndef TANH_LAYER_H
#define TANH_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Tanh : public Layer {
		public:
			Tanh();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class Tanh : virtual public CPU::Tanh {

		public:
			Tanh();


		};
	}
}

#endif

