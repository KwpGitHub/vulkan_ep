#ifndef QUANTIZE_LAYER_H
#define QUANTIZE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Quantize : public Layer {
		public:
			Quantize();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			float scale;
		};
	}
	namespace GPU {
		class Quantize : virtual public CPU::Quantize {

		public:
			Quantize();

		};
	}
}

#endif

