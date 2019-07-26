
#ifndef REDUCEL2_H
#define REDUCEL2_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL2 : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] axes;
		int keepdims;
    };
    vuh::Program<Specs, Params>* program;

    public:
       ReduceL2 (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/reducel2.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~ReduceL2 () {}
        

    };
}

#endif
