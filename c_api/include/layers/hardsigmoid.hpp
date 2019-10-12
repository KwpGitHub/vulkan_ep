#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Hardsigmoid : public layer
		{
		public:
			Hardsigmoid(float _alpha=1.0f, float _beta=1.0f);
			bool forward(tensor& in, tensor& out);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;		
			float m_alpha;
			float m_beta;	
		};
	}
}

#endif