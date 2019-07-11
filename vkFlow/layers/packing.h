#ifndef PACKING_LAYER_H
#define PACKING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Packing : public Layer {
		public:
			Packing();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int out_packing;
			int use_padding;
		};
	}
	namespace GPU {
		class Packing : virtual public CPU::Packing {

		public:
			Packing();

		};
	}
}

#endif

