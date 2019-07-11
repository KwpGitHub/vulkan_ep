#ifndef _LAYER_H
#define ABS_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Interp : public Layer {
		public:
			Interp();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			
			int resize_type; //1=nearest  2=bilinear  3=bicubic
			float width_scale;
			float height_scale;
			int output_width;
			int otput_height;
		};
	}
	namespace GPU {
		class Interp : virtual public CPU::Interp {

		public:
			Interp();

		};
	}
}

#endif

