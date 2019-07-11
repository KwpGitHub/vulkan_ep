#ifndef MEMORYDATA_LAYER_H
#define MEMORYDATA_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class MemoryData : public Layer {
		public:
			MemoryData();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int w, h, c;

			Mat data;
		};
	}
	namespace GPU {
		class MemoryData : virtual public CPU::MemoryData {

		public:
			MemoryData();


		};
	}
}

#endif

