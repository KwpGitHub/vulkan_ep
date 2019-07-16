#ifndef NORMALIZE_LAYER_H
#define NORMALIZE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Normalize : public Layer {
		public:
			Normalize();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int across_spatial;
			int across_channel;
			int channel_shared;
			float eps;
			int scale_data_size;
		
			Mat scale_data;
		};
	}
	namespace GPU {
		class Normalize : virtual public CPU::Normalize {

		public:
			Normalize();

		};
	}
}

#endif

