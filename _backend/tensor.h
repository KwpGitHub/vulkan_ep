#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <numeric>
#include <map>

#include "kernel/array.hpp"
#include "kernel/vuh.h"

namespace backend {
	inline vuh::Instance* g_instance;
	inline vuh::Device* g_device;
}

namespace backend {

	struct Shape_t {
		uint32_t n;
		uint32_t c;
		uint32_t d;
		uint32_t h;
		uint32_t w;
	};

	class Tensor {		
				
		size_t size;

	public:
		std::string name;
		vuh::Device* dev;
		vuh::Array<float>* data;
		Shape_t dims;

		Tensor(): data(nullptr), size(0u), dev(nullptr) {}
		
		Tensor(const std::vector<float>& d, Shape_t s): dims(s) {
			dev = g_device;
			size = (size_t)dims.n * (size_t)dims.c * (size_t)dims.d * (size_t)dims.h * (size_t)dims.w;
			data = new vuh::Array<float>(*dev, begin(d), end(d));
		}
		
		Tensor(const Tensor& t) {			
			data = t.data;			
			dims = t.dims;
			size = t.size;
			dev = t.dev;
		}

		std::vector<float> to_vector() {
			std::vector<float> t(size, 0.0);
			data->toHost(begin(t));
			return t;
		}

		void to(int d) {			
			std::vector<float> t(size, 0.0);
			data->toHost(begin(t));
			delete data;			
			data = new vuh::Array<float>(g_instance->devices().at(d), t);
		}

		void to(vuh::Device* d) {
			std::vector<float> t(size, 0.0);
			data->toHost(begin(t));
			delete data;
			data = new vuh::Array<float>(*d, t);
		}

		Shape_t shape() {
			return dims;
		}

		~Tensor() {
			delete data;			
		}

	};
}


namespace backend {
	inline std::map<std::string, Tensor*> tensor_dict;
	inline std::string file_path;
}

#endif