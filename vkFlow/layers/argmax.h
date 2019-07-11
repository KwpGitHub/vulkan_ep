#ifndef ARGMAX_LAYER_H
#define ARGMAX_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ArgMax : public Layer {
		public:
			ArgMax();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int out_max_val; int topk;
		};
	}
	namespace GPU {
		class ArgMax : virtual public CPU::ArgMax {

		public:
			ArgMax();
		};
	}
}

#endif

