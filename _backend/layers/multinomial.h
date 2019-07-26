
#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Multinomial : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int dtype;
		int sample_size;
		float seed;
    };
    vuh::Program<Specs, Params>* program;

    public:
       Multinomial (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/multinomial.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~Multinomial () {}
        

    };
}

#endif
