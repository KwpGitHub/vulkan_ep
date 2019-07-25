#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "kernel/vuh.h"

namespace backend {
	static vuh::Instance* instance;

	class Layer
	{
	public:
		Layer() {}
		virtual ~Layer() {}

	protected:
		vuh::Device* device;
		std::vector<float> input;
		std::vector<float>output;
		vuh::Array<float>* d_input;
		vuh::Array<float>* d_output;
	};
}

#endif //!LAYER_H