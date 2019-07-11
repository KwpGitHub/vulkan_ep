#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Pooling : public Layer {
		public:
			Pooling();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

			int pooling_type;
			int kernel_w, kernel_h; int stride_w, stride_h; int pad_left, pad_right, pad_top, pad_bottom;
			int global_pooling;
			int pad_mode; // 0=full 1=valid 2=SAME
		};
	}
	namespace GPU {
		class Pooling : virtual public CPU::Pooling {

		public:
			Pooling();

		};
	}
}

#endif

