#ifndef RNN_LAYER_H
#define RNN_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Rnn : public Layer {
		public:
			Rnn();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

			int num_output;
			int weight_data_size;

			Mat weight_hh_data;
			Mat weight_xh_data;
			Mat weight_ho_data;
			Mat bias_h_data;
			Mat bias_o_data;
		};
	}
	namespace GPU {
		class Rnn : virtual public CPU::Rnn {

		public:
			Rnn();

		};
	}
}

#endif

