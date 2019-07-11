#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Input : public Layer {
		public:
			Input();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int w, h, c;
		};
	}
	namespace GPU {
		class Input : virtual public CPU::Input {

		public:
			Input();

		};
	}
}

#endif

