#ifndef CLIP_LAYER_H
#define CLIP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU 
	{
		class Clip : public Layer {
		public:
			Clip();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float max; float min;
		};
	}
	namespace GPU {
		class Clip : virtual public CPU::Clip {

		public:
			Clip();

		};
	}
}

#endif

