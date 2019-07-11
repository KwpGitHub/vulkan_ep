#ifndef EMBED_LAYER_H
#define EMBED_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Embed : public Layer {
		public:
			Embed();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int num_output; int input_dim;
			int bias_term;
			int weight_data_size;

			Mat weight_data;
			Mat bias_data;
		};
	}
	namespace GPU {
		class Embed : virtual public CPU::Embed {

		public:
			Embed();
		};
	}
}

#endif

