#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Dropout : public Layer {
		public:
			Dropout();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float scale;
		};
	}
	namespace GPU {
		class Dropout : virtual public CPU::Dropout {

		public:
			Dropout();
		};
	}
}

#endif

