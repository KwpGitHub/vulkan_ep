#ifndef CONV1_H
#define CONV1_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Conv1d : public layer
		{
		public:
			Conv1d();
			bool forward(tensor& x, tensor& w, tensor& b, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef CONV2_H
#define CONV2_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Conv2d : public layer
		{
		public:
			Conv2d();
			bool forward(tensor& x, tensor& w, tensor& b, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef CONV3_H
#define CONV3_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class Conv3d : public layer
		{
		public:
			Conv3d();
			bool forward(tensor& x, tensor& w, tensor& b, tensor& y);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif