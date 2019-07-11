#ifndef ELTWISE_LAYER_H
#define ELTWISE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Eltwise : public Layer {
		public:
			Eltwise();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			enum { Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };

			int op_type;

			Mat coeffs;
		};
	}
	namespace GPU {
		class Eltwise : virtual public CPU::Eltwise {

		public:
			Eltwise();
		};
	}
}

#endif

