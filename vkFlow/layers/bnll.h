#ifndef BNLL_LAYER_H
#define BNLL_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class BNLL : public Layer {
		public:
			BNLL();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class BNLL : virtual public CPU::BNLL {

		public:
			BNLL();

		};
	}
}

#endif

