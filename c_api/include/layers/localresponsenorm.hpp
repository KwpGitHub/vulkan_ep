#ifndef LOCALRESPONSENORM1_H
#define LOCALRESPONSENORM1_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class LocalResponseNorm1d : public layer
		{
		public:
			LocalResponseNorm1d();
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef LOCALRESPONSENORM2_H
#define LOCALRESPONSENORM2_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class LocalResponseNorm2d : public layer
		{
		public:
			LocalResponseNorm2d();
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef LOCALRESPONSENORM3_H
#define LOCALRESPONSENORM3_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class LocalResponseNorm3d : public layer
		{
		public:
			LocalResponseNorm3d();
			bool forward(tensor& x, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif