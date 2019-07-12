#ifndef ROALIGN_LAYER_H
#define ROALIGN_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ROIAlign : public Layer {
		public:
			ROIAlign();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int pooled_width, pooled_height;
			float spatial_scale;
		};
	}
	namespace GPU {
		class ROIAlign : virtual public CPU::ROIAlign {

		public:
			ROIAlign();

		};
	}
}

#endif

