#ifndef CAST_LAYER_H
#define CAST_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Cast : public Layer {
		public:
			Cast();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int type_from;	int type_to;
		};
	}
	namespace GPU {
		class Cast : virtual public CPU::Cast {

		public:
			Cast();

			virtual int create_pipeline(const Option& opt);
			virtual int destroy_pipeline(const Option& opt);
			virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

			Pipeline* pipeline_absval;
			Pipeline* pipeline_absval_pack4;

		};
	}
}

#endif

