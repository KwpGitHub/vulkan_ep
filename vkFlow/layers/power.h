#ifndef POWER_LAYER_H
#define POWER_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Power : public Layer {
		public:
			Power();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float power, scale, shift;
		};
	}
	namespace GPU {
		class Power : virtual public CPU::Power {

		public:
			Power();


		};
	}
}

#endif

