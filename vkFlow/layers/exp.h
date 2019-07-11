#ifndef EXP_LAYER_H
#define EXP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Exp : public Layer {
		public:
			Exp();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float base;
			float scale;
			float shift;
		};
	}
	namespace GPU {
		class Exp : virtual public CPU::Exp {

		public:
			Exp();
		};
	}
}

#endif

