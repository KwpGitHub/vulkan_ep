#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class BatchNorm : public Layer {
		public:
			BatchNorm();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
			
			int channels; float eps;

			Mat slope_data;
			Mat mean_data;
			Mat var_data;
			Mat bias_data;
			Mat a_data;
			Mat b_data;

		};
	}
	namespace GPU {
		class BatchNorm : virtual public CPU::BatchNorm {

		public:
			BatchNorm();
		};
	}
}

#endif

