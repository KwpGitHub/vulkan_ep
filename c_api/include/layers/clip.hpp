#ifndef CLIP_H
#define CLIP_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Clip : public layer
		{
		public:
			Clip(float _max = 1.0f, float _min = 0.0f);
			bool forward(tensor& in, tensor& out);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;	
			float m_max;
			float m_min;		
		};
	}
}

#endif