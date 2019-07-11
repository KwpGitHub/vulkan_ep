#ifndef SELU_LAYER_H
#define SELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Selu : public Layer {
		public:
			Selu();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float alpha, lambda;
		};
	}
	namespace GPU {
		class Selu : virtual public CPU::Selu {

		public:
			Selu();


		};
	}
}

#endif

