
#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LinearRegressor : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		float[] coefficients;
		float[] intercepts;
		std::string post_transform;
		int targets;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LinearRegressor (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/linearregressor.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LinearRegressor () {}

    };
}

#endif
