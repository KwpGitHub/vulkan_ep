#ifndef SQUEEZE_LAYER_H
#define SQUEEZE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Squeeze : public Layer {
		public:
			Squeeze();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int squeeze_w, squeeze_h, squeeze_c;
		};
	}
	namespace GPU {
		class Squeeze : virtual public CPU::Squeeze {

		public:
			Squeeze();

		};
	}
}

#endif

