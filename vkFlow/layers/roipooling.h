#ifndef ROIPOOLING_LAYER_H
#define ROIPOOLING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ROIPooling : public Layer {
		public:
			ROIPooling();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			
			int pooled_width, pooled_height;
			float spatial_scale;
		};
	}
	namespace GPU {
		class ROIPooling : virtual public CPU::ROIPooling {

		public:
			ROIPooling();

		};
	}
}

#endif

