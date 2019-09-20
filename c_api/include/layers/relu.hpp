#ifndef RELU_H
#define RELU_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Relu : public layer
		{
		public:
			Relu();
			bool forward(tensor& in, tensor& out);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
			float m_slope;
		};
	}
}

#endif