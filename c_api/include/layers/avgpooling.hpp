#ifndef AVGPOOLING1_H
#define AVGPOOLING1_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class AvgPooling1d : public layer
		{
		public:
			AvgPooling1d();
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

#ifndef AVGPOOLING2_H
#define AVGPOOLING2_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class AvgPooling2d : public layer
		{
		public:
			AvgPooling2d();
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

#ifndef AVGPOOLING3_H
#define AVGPOOLING3_H

#include "kernel/kernel.hpp"
#include "kernel/layer.hpp"

namespace kernel {
	namespace layers {
		class AvgPooling3d : public layer
		{
		public:
			AvgPooling3d();
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