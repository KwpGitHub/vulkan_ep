#ifndef DETECTIONOUTPUT_LAYER_H
#define DETECTIONOUTPUT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class DetectionOutput : public Layer {
		public:
			DetectionOutput();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int num_class;
			float nms_threshold;
			int nms_top_k;
			int keep_top_k;
			float confidence_threshold;
			float variances[4];
		};
	}
	namespace GPU {
		class DetectionOutput : virtual public CPU::DetectionOutput {

		public:
			DetectionOutput();
		};
	}
}

#endif

