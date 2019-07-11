#ifndef SHUFFLECHANNEL_LAYER_H
#define SHUFFLECHANNEL_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class ShuffleChannel : public Layer {
		public:
			ShuffleChannel();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int group;
		};
	}
	namespace GPU {
		class ShuffleChannel : virtual public CPU::ShuffleChannel {

		public:
			ShuffleChannel();


		};
	}
}

#endif

