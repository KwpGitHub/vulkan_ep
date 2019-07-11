#ifndef PRELU_LAYER_H
#define PRELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class PRelu : public Layer {
		public:
			PRelu();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int num_slope;

			Mat slope_data;
		};
	}
	namespace GPU {
		class PRelu : virtual public CPU::PRelu {

		public:
			PRelu();


		};
	}
}

#endif

