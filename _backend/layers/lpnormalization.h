
#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpNormalization : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int axis;
		int p;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LpNormalization (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/lpnormalization.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LpNormalization () {}

    };
}

#endif
