#ifndef BNLL_LAYER_H
#define BNLL_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class bnll : public Layer {
		public:
			bnll();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class bnll : virtual public CPU::bnll {

		public:
			bnll();

		};
	}
}

#endif

