#ifndef HARDSIGMOID_LAYER_H
#define HARDSIGMOID_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class HardSigmoid : public Layer {
		public:
			HardSigmoid();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float alpha, beta, lower, upper;
		};
	}
	namespace GPU {
		class HardSigmoid : virtual public CPU::HardSigmoid {

		public:
			HardSigmoid();


		};
	}
}