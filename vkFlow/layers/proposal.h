#ifndef PROPOSAL_LAYER_H
#define PROPOSAL_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Proposal : public Layer {
		public:
			Proposal();
			virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
			
			int feat_stride;
			int base_size;
			int pre_nms_topN;
			int after_nms_topN;
			float nms_thresh;
			int min_size;

			Mat ratios;
			Mat scales;
			Mat anchors;
		};
	}
	namespace GPU {
		class Proposal : virtual public CPU::Proposal {

		public:
			Proposal();


		};
	}
}

#endif

