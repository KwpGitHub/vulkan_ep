#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Concat : public Layer {
		public:
			Concat();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int axis;
		};
	}
	namespace GPU {
		class Concat : virtual public CPU::Concat {

		public:
			Concat();

			

		};
	}
}

#endif

