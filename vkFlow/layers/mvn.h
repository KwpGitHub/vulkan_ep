#ifndef MVN_LAYER_H
#define ABS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Mvn : public Layer {
		public:
			Mvn();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int normalize_variance;
			int across_channels;
			float eps;
		};
	}
	namespace GPU {
		class Mvn : virtual public CPU::Mvn {

		public:
			Mvn();


		};
	}
}

#endif

