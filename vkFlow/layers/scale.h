#ifndef SCALE_LAYER_H
#define SCALE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Scale : public Layer {
		public:
			Scale();
			virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			int scale_data_size;
			int bias_term;

			Mat scale_data;
			Mat bias_data;
		};
	}
	namespace GPU {
		class Scale : virtual public CPU::Scale {

		public:
			Scale();

		};
	}
}

#endif

