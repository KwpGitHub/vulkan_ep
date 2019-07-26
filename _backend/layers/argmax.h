
#ifndef ARGMAX_H
#define ARGMAX_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMax : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int axis;
		int keepdims;
    };
    vuh::Program<Specs, Params>* program;

    public:
       ArgMax (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/argmax.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~ArgMax () {}
        

    };
}

#endif
