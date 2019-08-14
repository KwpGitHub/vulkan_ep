#ifndef LAYER_H
#define LAYER_H

#include <map>
#include <vector>

#include "kernel/vuh.h"
#include "tensor.h"

namespace backend {
	static char static_execution = 0;

#define PROCESSKERNEL_SIZE 32
	
	class Layer
	{
	public:
		Layer(std::string n) : name(n) {}
		virtual ~Layer() {}
		virtual void forward() {}
	protected:
		//virtual void parameter_proc(std::map<std::string, std::vector<std::string>> a()) {}
		std::vector<std::string> inputs;
		std::vector<std::string> outputs;
		std::string name;
		using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
	};
}

namespace backend {
	std::map<std::string, Layer*> layer_dict;
}


#endif //!LAYER_H