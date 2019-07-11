#ifndef THRESHOLD_LAYER_H
#define THRESHOLD_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Threshold : public Layer {
		public:
			Threshold();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float threshold;
		};
	}
	namespace GPU {
		class Threshold : virtual public CPU::Threshold {

		public:
			Threshold();

		};
	}
}

#endif

