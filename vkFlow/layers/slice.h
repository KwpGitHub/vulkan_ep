#ifndef SLICE_LAYER_H
#define SLICE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Slice : public Layer {
		public:
			Slice();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int axis;

			Mat slices;
		};
	}
	namespace GPU {
		class Slice : virtual public CPU::Slice {

		public:
			Slice();

		};
	}
}

#endif

