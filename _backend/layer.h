#pragma once
#include "tensor.h"
#ifndef LAYER_H
#define LAYER_H
#include <map>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "kernel/vuh.h"

namespace backend {
	
	

#define PROCESSKERNEL_SIZE 32
	
	class Layer
	{
	public:
		std::string name;
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
	};
}

namespace backend {
	static std::map<std::string, Layer*> layer_dict;
	template<typename T> std::vector<T> convert(py::list l) {
		std::vector<T> x;
		for (auto i : l)
			x.push_back(i.cast<T>());
		return x;
	}
	static char static_execution = 0;
	static py::module nn;
}


#endif //!LAYER_H