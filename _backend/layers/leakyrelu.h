
#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LeakyRelu : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float alpha;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LeakyRelu (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/leakyrelu.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LeakyRelu () {}
        

    };
}

#endif
