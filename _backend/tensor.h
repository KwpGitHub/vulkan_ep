
#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include "kernel/vuh.h"
namespace backend {
	static vuh::Instance* instance;

	class Tensor
	{
		vuh::Device* dev;
		vuh::Array<float>* d_x;
		std::vector<float> x;
		std::vector<uint32_t> dims;

	public:
		Tensor() : d_x(nullptr) {
			dev = new vuh::Device(instance->devices().at(0));
		}
		
		Tensor(const std::vector<float>& d, const std::vector<uint32_t> s) : x(d), dims(s) {
			dev = new vuh::Device(instance->devices().at(0));
			d_x = new vuh::Array<float>(*dev, x);
		}
		
		Tensor(const Tensor& t) {
			dev = t.dev;
			d_x = t.d_x;
			x = t.x;
			dims = t.dims;
		}

		std::vector<float>& to_vector() {
			auto t = x;
			d_x->toHost(begin(t));
			return t;
		}

		vuh::Array<float>* data() {
			return d_x;
		}

		void to(int d) {			
			auto ndev = new vuh::Device(instance->devices().at(d));
			d_x->toHost(begin(x));
			delete d_x;
			delete dev;
			dev = ndev;
			d_x = new vuh::Array<float>(*dev, x);
		}

		void to(vuh::Device* d) {
			d_x->toHost(begin(x));
			delete d_x;
			delete dev;
			dev = d;
			d_x = new vuh::Array<float>(*dev, x);
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



#endif