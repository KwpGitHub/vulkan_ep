#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "kernel/vuh.h"
#include "tensor.h"

namespace backend {
	static vuh::Instance* instance;
	static char static_execution = 0;

#define PROCESSKERNEL_SIZE 32

	
	class Layer
	{
	public:
		Layer() {}
		virtual ~Layer() {}
		virtual void build_pipeline(){}
		virtual void forward() {}
	protected:
		vuh::Device* device;
		Tensor input;
		Tensor output;
		
	};
}

#endif //!LAYER_H