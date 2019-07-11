#ifndef ROALIGN_LAYER_H
#define ROALIGN_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class RoAlign : public Layer {
		public:
			RoAlign();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int pooled_width, pooled_height;
			float spatial_scale;
		};
	}
	namespace GPU {
		class RoAlign : virtual public CPU::RoAlign {

		public:
			RoAlign();

		};
	}
}

#endif

