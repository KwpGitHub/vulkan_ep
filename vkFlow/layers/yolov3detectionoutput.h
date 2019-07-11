#ifndef ABS_LAYER_H
#define ABS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class AbsVal : public Layer {
		public:
			AbsVal();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
		};
	}
	namespace GPU {
		class AbsVal : virtual public CPU::AbsVal {

		public:
			AbsVal();

			virtual int create_pipeline(const Option& opt);
			virtual int destroy_pipeline(const Option& opt);
			virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

			Pipeline* pipeline_absval;
			Pipeline* pipeline_absval_pack4;

		};
	}
}

#endif

