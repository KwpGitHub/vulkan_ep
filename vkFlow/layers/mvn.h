#ifndef MVN_LAYER_H
#define ABS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class MVN : public Layer {
		public:
			MVN();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int normalize_variance;
			int across_channels;
			float eps;
		};
	}
	namespace GPU {
		class MVN : virtual public CPU::MVN {

		public:
			MVN();


		};
	}
}

#endif

