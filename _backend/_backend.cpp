#include <pybind11/pybind11.h>
#include <vector>
#include "layer.h"
#include "layers.h"
#include "kernel/vuh.h"

/*
class ABS : public Layer {
	
	using Specs = vuh::typelist<uint32_t>;
	struct Params { uint32_t size; float a; };

	vuh::Program<Specs, Params>* program;
public:
	ABS(const std::vector<float> &tinput) {
		input = tinput;
		output = std::vector<float>(128, 0.0f);
		device =  new vuh::Device(backend::instance->devices().at(0));
		program = new vuh::Program<Specs, Params>(*device, "./shaders/abs.spv");
		d_input = new vuh::Array<float>(*device, input);
		d_output = new vuh::Array<float>(*device, output);
		
	}
	~ABS(){}

	void run() {
		program->grid(128 / 64).spec(64)({ 128, 0.1f }, *d_output, *d_input);
		d_output->toHost(begin(output));
		printf("DONE");
	}
private:

};
*/

void create_instance() {
	backend::instance = new vuh::Instance();
	
}


PYBIND11_MODULE(_backend, m) {
	m.def("create_instance", &create_instance);
}