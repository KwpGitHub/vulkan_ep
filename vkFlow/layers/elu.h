#ifndef ELU_LAYER_H
#define ELU_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Elu : public Layer {
		public:
			Elu();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float alpha;
		};
	}
	namespace GPU {
		class Elu : virtual public CPU::Elu {

		public:
			Elu();

		};
	}
}

#endif

