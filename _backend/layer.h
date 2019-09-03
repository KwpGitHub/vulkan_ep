#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <map>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "tensor.h"
#include "kernel/vuh.h"

namespace backend {
	
#define PROCESSKERNEL_SIZE_x 1
#define PROCESSKERNEL_SIZE_y 1
#define PROCESSKERNEL_SIZE_z 1
	
	class Layer
	{
	public:
		std::string name;
		Layer(std::string name) : name(name) {}
		virtual ~Layer() {}
		virtual void forward() {}
		virtual void init() {}
		virtual void bind() {}
		virtual void build() {}
	protected:
		using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
	};
}

namespace backend {
	inline std::map<std::string, Layer*> layer_dict;

	template<typename T> std::vector<T> convert(py::list l) {
		std::vector<T> x;
		for (auto i : l)
			x.push_back(i.cast<T>());
		return x;
	}

}

#endif //!LAYER_H