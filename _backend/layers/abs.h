
#ifndef ABS_H
#define ABS_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Abs : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

    };
    vuh::Program<Specs, Params>* program;

    public:
       Abs (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/abs.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~Abs () {}

    };
}

#endif
