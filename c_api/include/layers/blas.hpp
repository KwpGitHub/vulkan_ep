#ifndef GEMM_H
#define GEMM_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Gemm : public layer
		{
		public:
			Gemm();
			bool forward(tensor& in, tensor& out);
			void reshapeOutTensor(tensor& A, tensor& B, tensor& C tensor& D);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;	
            int M;
            int N;
            int K;
            bool use_bias;		
		};
	}
}

#endif