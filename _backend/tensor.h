#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <map>

#include "kernel/vuh.h"

namespace backend {
	static vuh::Instance* instance;

	class Tensor
	{
		vuh::Device* dev;
		vuh::Array<float>* d_x;		
		std::vector<uint32_t> dims;

	public:

		std::string name;

		Tensor() : d_x(nullptr) {
			dev = new vuh::Device(instance->devices().at(0));
		}
		
		Tensor(const std::vector<float>& d, const std::vector<uint32_t> s): dims(s) {
			dev = new vuh::Device(instance->devices().at(0));
			d_x = new vuh::Array<float>(*dev, d);
		}
		
		Tensor(const Tensor& t) {
			dev = t.dev;
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
			auto ndev = new vuh::Device(instance->devices().at(d));
			std::vector<float> t(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0.0);
			d_x->toHost(begin(t));
			delete d_x;
			delete dev;
			dev = ndev;
			d_x = new vuh::Array<float>(*dev, t);
		}

		void to(vuh::Device* d) {
			std::vector<float> t(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0.0);
			d_x->toHost(begin(t));
			delete d_x;
			delete dev;
			dev = d;
			d_x = new vuh::Array<float>(*dev, t);
		}

		std::vector<uint32_t> shape() {
			return dims;
		}

		~Tensor() {
			delete d_x;
			delete dev;
		}

	};
}

namespace backend {
	static std::map<std::string, Tensor*> tensor_dict;
}

#endif