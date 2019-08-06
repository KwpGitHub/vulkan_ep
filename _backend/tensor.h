#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <map>

#include "kernel/array.hpp"
#include "kernel/vuh.h"

namespace backend {

	struct Shape_t {
		uint32_t n;
		uint32_t c;
		uint32_t d;
		uint32_t h;
		uint32_t w;
	};

	static vuh::Instance* instance;
	static vuh::Device* device;
	static std::string file_path;
	class Tensor {		
		
		vuh::Array<float>* d_x;		
		size_t size;

	public:
		std::string name;
		vuh::Device* dev;
		Shape_t dims;

		Tensor(): d_x(nullptr), size(0u), dev(nullptr) {}
		
		Tensor(const std::vector<float>& d, Shape_t s): dims(s) {
			dev = device;
			size = (size_t)dims.n * (size_t)dims.c * (size_t)dims.d * (size_t)dims.h * (size_t)dims.w;
			d_x = new vuh::Array<float>(*dev, begin(d), end(d));
		}
		
		Tensor(const Tensor& t) {			
			d_x = t.d_x;			
			dims = t.dims;
			size = t.size;
			dev = t.dev;
		}

		std::vector<float>& to_vector() {
			std::vector<float> t(size, 0.0);
			d_x->toHost(begin(t));
			return t;
		}

		vuh::Array<float>* data() {
			return d_x;
		}

		void to(int d) {			
			std::vector<float> t(size, 0.0);
			d_x->toHost(begin(t));
			delete d_x;			
			d_x = new vuh::Array<float>(instance->devices().at(d), t);
		}

		void to(vuh::Device* d) {
			std::vector<float> t(size, 0.0);
			d_x->toHost(begin(t));
			delete d_x;
			d_x = new vuh::Array<float>(*d, t);
		}

		Shape_t shape() {
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