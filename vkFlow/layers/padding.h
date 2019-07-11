#ifndef PADDING_LAYER_H
#define PADDING_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Padding : public Layer {
		public:
			Padding();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int top, bottom, left, right;
			int type;// 0=BORDER_CONSTANT 1=BORDER_REPLICATE
			float value;
		};
	}
	namespace GPU {
		class Padding : virtual public CPU::Padding {

		public:
			Padding();

		};
	}
}

#endif

