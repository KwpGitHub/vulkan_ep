#ifndef REDUCTION_LAYER_H
#define REDUCTION_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Reduction : public Layer {
		public:
			Reduction();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			enum {
				ReductionOp_SUM = 0,
				ReductionOp_ASUM = 1,
				ReductionOp_SUMSQ = 2,
				ReductionOp_MEAN = 3,
				ReductionOp_MAX = 4,
				ReductionOp_MIN = 5,
				ReductionOp_PROD = 6
			};

			int operation;
			int dim;
			float coeff;
		};
	}
	namespace GPU {
		class Reduction : virtual public CPU::Reduction {

		public:
			Reduction();


		};
	}
}

#endif

