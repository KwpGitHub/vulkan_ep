
#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class HardSigmoid : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float alpha;
		float beta;
    };
    vuh::Program<Specs, Params>* program;

    public:
       HardSigmoid (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/hardsigmoid.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~HardSigmoid () {}
        

    };
}

#endif
