
#ifndef ARGMIN_H
#define ARGMIN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMin : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int axis;
		int keepdims;
    };
    vuh::Program<Specs, Params>* program;

    public:
       ArgMin (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/argmin.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~ArgMin () {}

    };
}

#endif
