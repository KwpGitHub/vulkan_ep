
#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class InstanceNormalization : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float epsilon;
    };
    vuh::Program<Specs, Params>* program;

    public:
       InstanceNormalization (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/instancenormalization.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~InstanceNormalization () {}
        

    };
}

#endif
