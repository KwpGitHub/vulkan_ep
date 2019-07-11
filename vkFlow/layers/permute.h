#ifndef PERMUTE_LAYER_H
#define PERMUTE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Permute : public Layer {
		public:
			Permute();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int order_type;
		};
	}
	namespace GPU {
		class Permute : virtual public CPU::Permute {

		public:
			Permute();

		};
	}
}

#endif

