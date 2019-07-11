#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Flatten : public Layer {
		public:
			Flatten();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class Flatten : virtual public CPU::Flatten {

		public:
			Flatten();

		};
	}
}

#endif

