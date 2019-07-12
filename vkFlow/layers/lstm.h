#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class LSTM : public Layer {
		public:
			LSTM();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			
			int num_output;
			int weight_data_size;

			Mat weight_hc_data;
			Mat weight_xc_data;
			Mat bias_c_data;			
		};
	}
	namespace GPU {
		class LSTM : virtual public CPU::LSTM {

		public:
			LSTM();
		};
	}
}

#endif

