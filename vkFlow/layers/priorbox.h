#ifndef PRIORBOX_LAYER_H
#define PRIORBOX_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class PriorBox : public Layer {
		public:
			PriorBox();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			Mat min_sizes;
			Mat max_sizes;
			Mat aspect_ratios;

			float variances[4];
			int flip, clip;
			int image_width, image_height;
			float step_width, step_height;
			float offset;
		};
	}
	namespace GPU {
		class PriorBox : virtual public CPU::PriorBox {

		public:
			PriorBox();


		};
	}
}

#endif

