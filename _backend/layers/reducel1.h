
#ifndef REDUCEL1_H
#define REDUCEL1_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL1 : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int[] axes;
		int keepdims;
    };
    vuh::Program<Specs, Params>* program;

    public:
       ReduceL1 (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/reducel1.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~ReduceL1 () {}

    };
}

#endif
