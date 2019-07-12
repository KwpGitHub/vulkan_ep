#ifndef PRELU_LAYER_H
#define PRELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class PReLU : public Layer {
		public:
			PReLU();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int num_slope;

			Mat slope_data;
		};
	}
	namespace GPU {
		class PReLU : virtual public CPU::PReLU {

		public:
			PReLU();


		};
	}
}

#endif

