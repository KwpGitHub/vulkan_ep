#ifndef LRN_LAYER_H
#define LRN_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Lrn : public Layer {
		public:
			Lrn();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
			enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };

			int region_type;
			int local_size;
			int alpha, beta, bias;
		};
	}
	namespace GPU {
		class Lrn : virtual public CPU::Lrn {
		public:
			Lrn();
		};
	}
}

#endif

