#ifndef PSROIPOOLING_LAYER_H
#define PSROIPOOLING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class PSROIPooling : public Layer {
		public:
			PSROIPooling();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int pooled_width, pooled_height;
			float spatial_scale;
			int output_dim;
		};
	}
	namespace GPU {
		class PSROIPooling : virtual public CPU::PSROIPooling {

		public:
			PSROIPooling();


		};
	}
}

#endif

