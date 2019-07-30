#ifndef LAYER_H
#define LAYER_H
#include <vector>

#include "kernel/vuh.h"
#include "tensor.h"

namespace backend {
	static char static_execution = 0;

#define PROCESSKERNEL_SIZE 32
	
	class Layer
	{
	public:
		Layer() {}
		virtual Tensor& operator() (const Tensor& it)  { return &it; }
		virtual ~Layer() {}
		virtual void build_pipeline(){}
		virtual void forward() {}
	protected:

		Tensor input;
	};
}


#endif //!LAYER_H