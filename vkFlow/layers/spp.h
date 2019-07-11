#ifndef SPP_LAYER_H
#define SPP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Spp : public Layer {
		public:
			Spp();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

			int pooling_type;
			int pyramid_height;
		};
	}
	namespace GPU {
		class Spp : virtual public CPU::Spp {

		public:
			Spp();

		};
	}
}

#endif

