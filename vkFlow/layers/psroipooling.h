#ifndef PSROIPOOLING_LAYER_H
#define PSROIPOOLING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class PsroiPooling : public Layer {
		public:
			PsroiPooling();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int pooleed_width, pooled_height;
			float spatial_scale;
			int output_dim;
		};
	}
	namespace GPU {
		class PsroiPooling : virtual public CPU::PsroiPooling {

		public:
			PsroiPooling();


		};
	}
}

#endif

