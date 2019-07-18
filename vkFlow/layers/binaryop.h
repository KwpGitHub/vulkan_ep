#ifndef BINOP_LAYER_H
#define BINOP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class BinaryOP : public Layer {
		public:
			BinaryOP();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
			enum {
				Operation_ADD = 0,
				Operation_SUB = 1,
				Operation_MUL = 2,
				Operation_DIV = 3,
				Operation_MAX = 4,
				Operation_MIN = 5,
				Operation_POW = 6,
				Operation_RSUB = 7,
				Operation_RDIV = 8
			};
			
			int op_type;
			int with_scalar;
			float b;		
		};
	}
	namespace GPU {
		class BinaryOP : virtual public CPU::BinaryOP {

		public:
			BinaryOP();

		};
	}
}

#endif

