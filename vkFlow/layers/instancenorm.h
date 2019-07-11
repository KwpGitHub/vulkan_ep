#ifndef INSTANCENORM_LAYER_H
#define INSTANCENORM_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class InstanceNorm : public Layer {
		public:
			InstanceNorm();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int channels;
			float eps;

			Mat gamma_data;
			Mat beta_data;
		};
	}
	namespace GPU {
		class InstanceNorm : virtual public CPU::InstanceNorm {

		public:
			InstanceNorm();
		};
	}
}

#endif

