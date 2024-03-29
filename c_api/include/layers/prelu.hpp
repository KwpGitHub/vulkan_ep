#ifndef PRELU_H
#define PRELU_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class PReLU : public layer
		{
		public:
			PReLU(float _slope=0.2f);
			bool forward(tensor& in, tensor& out);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
			float m_slope;
		};
	}
}

#endif