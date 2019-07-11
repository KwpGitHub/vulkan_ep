#ifndef BINOP_LAYER_H
#define ABS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class bin_op : public Layer {
		public:
			bin_op();
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
		class bin_op : virtual public CPU::bin_op {

		public:
			bin_op();

			virtual int create_pipeline(const Option& opt);
			virtual int destroy_pipeline(const Option& opt);
			virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

			Pipeline* pipeline_absval;
			Pipeline* pipeline_absval_pack4;

		};
	}
}

#endif

