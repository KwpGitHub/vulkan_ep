#ifndef UNARYOP_LAYER_H
#define UNARYOP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class UnaryOP : public Layer {
		public:
			UnaryOP();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
			enum {
				Operation_ABS = 0,
				Operation_NEG = 1,
				Operation_FLOOR = 2,
				Operation_CEIL = 3,
				Operation_SQUARE = 4,
				Operation_SQRT = 5,
				Operation_RSQRT = 6,
				Operation_EXP = 7,
				Operation_LOG = 8,
				Operation_SIN = 9,
				Operation_COS = 10,
				Operation_TAN = 11,
				Operation_ASIN = 12,
				Operation_ACOS = 13,
				Operation_ATAN = 14,
				Operation_RECIPROCAL = 15
			};

			int op_type;
		};
	}
	namespace GPU {
		class UnaryOP : virtual public CPU::UnaryOP {

		public:
			UnaryOP();
		};
	}
}

#endif

