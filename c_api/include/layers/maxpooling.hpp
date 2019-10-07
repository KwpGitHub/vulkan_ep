#ifndef MAXPOOLING1_H
#define MAXPOOLING1_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class MaxPooling1d : public layer
		{
		public:
			MaxPooling1d();
			bool forward(tensor& x, tensor& y, tensor& mask);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef MAXPOOLING2_H
#define MAXPOOLING2_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class MaxPooling2d : public layer
		{
		public:
			MaxPooling2d();
			bool forward(tensor& x, tensor& y, tensor& mask);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif

#ifndef MAXPOOLING3_H
#define MAXPOOLING3_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class MaxPooling3d : public layer
		{
		public:
			MaxPooling3d();
			bool forward(tensor& x, tensor& y, tensor& mask);
			void reshapeOutTensor(tensor& in, tensor& out);
			virtual bool forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs);
		private:
			bool computeGroupCount();
			int m_total;
		};
	}
}

#endif