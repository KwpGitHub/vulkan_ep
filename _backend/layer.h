#pragma once
#include "tensor.h"
#ifndef LAYER_H
#define LAYER_H
#include <map>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "kernel/vuh.h"

namespace backend {
	
	static char static_execution = 0;
	static py::module nn;
	
#define PROCESSKERNEL_SIZE 32
	
	class Layer
	{
	public:
		Layer(std::string name) : name(name) {}
		virtual ~Layer() {}
		//virtual void forward() {}
		virtual void init() {}
		virtual void bind() {}
		virtual void build() {}
	protected:
		//virtual void parameter_proc(std::map<std::string, std::vector<std::string>> a()) {}
		std::vector<std::string> inputs;
		std::vector<std::string> outputs;
		std::string name;
		using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
		
	};
}

namespace backend {
	template<typename T> Layer* createInstance(std::string n) { { return new T(n); } }
	static std::map<std::string, Layer*> layer_dict;
}


#endif //!LAYER_H