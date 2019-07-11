#ifndef BIAS_LAYER_H
#define BIAS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Bias : public Layer {
		public:
			Bias();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int bias_data_size;

			Mat bias_data;
		};
	}
	namespace GPU {
		class Bias : virtual public CPU::Bias {

		public:
			Bias();

		};
	}
}

#endif

