
#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Unsqueeze : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] axes;
    };
    vuh::Program<Specs, Params>* program;

    public:
       Unsqueeze (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/unsqueeze.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~Unsqueeze () {}
        

    };
}

#endif
