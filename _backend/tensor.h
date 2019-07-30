#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <map>

#include "kernel/array.hpp"
#include "kernel/vuh.h"

namespace backend {
	static vuh::Instance* instance;
	static vuh::Device* device;

	class Tensor {		
		vuh::Device* dev;
		vuh::Array<float, vuh::mem::Device>* d_x;

		std::vector<uint32_t> dims;
		size_t size;
	public:
		std::string name;

		Tensor(): d_x(nullptr), size(0u), dev(nullptr) {}
		
		Tensor(const std::vector<float>& d, const std::vector<uint32_t> s): dims(s) {
			dev = device;
			size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
			d_x = new vuh::Array<float, vuh::mem::Device>(*dev, begin(d), end(d));
		}
		
		Tensor(const Tensor& t) {			
			d_x = t.d_x;			
			dims = t.dims;
		}

		std::vector<float>& to_vector() {
			std::vector<float> t(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0.0);
			d_x->toHost(begin(t));
			return t;
		}

		vuh::Array<float>* data() {
			return d_x;
		}

		void to(int d) {			
			std::vector<float> t(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0.0);
			d_x->toHost(begin(t));
			delete d_x;			
			d_x = new vuh::Array<float>(instance->devices().at(d), t);
		}

		void to(vuh::Device* d) {
			std::vector<float> t(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0.0);
			d_x->toHost(begin(t));
			delete d_x;
			d_x = new vuh::Array<float>(*d, t);
		}

		std::vector<uint32_t> shape() {
			return dims;
		}

		~Tensor() {
			delete d_x;			
		}

	};
}

namespace backend {
	static std::map<std::string, Tensor*> tensor_dict;
}

#endif