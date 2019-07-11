#ifndef LOG_LAYER_H
#define LOG_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Log : public Layer {
		public:
			Log();
			virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

			float base, scale, shift;
		};
	}
	namespace GPU {
		class Log : virtual public CPU::Log {

		public:
			Log();
			
		};
	}
}

#endif

