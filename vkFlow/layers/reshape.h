#ifndef RESHAPE_LAYER_H
#define RESHAPE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Reshape : public Layer {
		public:
			Reshape();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int w, h, c;
			int permute;
			int ndim;
		};
	}
	namespace GPU {
		class Reshape : virtual public CPU::Reshape {

		public:
			Reshape();

		};
	}
}

#endif

