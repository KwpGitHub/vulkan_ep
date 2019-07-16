#ifndef EXPANDDIMS_LAYER_H
#define EXPENDDIMS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ExpandDim : public Layer {
		public:
			ExpandDim();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int expand_w;
			int expand_h;
			int expand_c;

		};
	}
	namespace GPU {
		class ExpandDim : virtual public CPU::ExpandDim {

		public:
			ExpandDim();
		};
	}
}

#endif

