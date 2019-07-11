#ifndef REORG_LAYER_H
#define REORG_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Reorg : public Layer {
		public:
			Reorg();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			
			int stride;
		};
	}
	namespace GPU {
		class Reorg : virtual public CPU::Reorg {

		public:
			Reorg();

		};
	}
}

#endif

