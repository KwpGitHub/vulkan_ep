
#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniform : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int dtype;
		float high;
		float low;
		float seed;
		int[] shape;
    };
    vuh::Program<Specs, Params>* program;

    public:
       RandomUniform (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/randomuniform.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~RandomUniform () {}

    };
}

#endif
