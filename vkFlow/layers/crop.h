#ifndef CROP_LAYER_H
#define CROP_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Crop : public Crop {
		public:
			Crop();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int woffset, hoffset, coffset;
			int outw, outh, outc;
		};
	}
	namespace GPU {
		class Crop : virtual public CPU::Crop 
		{
		public:
			Crop();

		};
	}
}

#endif

