#ifndef SPLIT_LAYER_H
#define SPLIT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Split : public Layer {
		public:
			Split();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
		};
	}
	namespace GPU {
		class Split : virtual public CPU::Split {

		public:
			Split();


		};
	}
}

#endif

