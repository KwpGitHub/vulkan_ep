
#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniformLike : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int dtype;
		float high;
		float low;
		float seed;
    };
    vuh::Program<Specs, Params>* program;

    public:
       RandomUniformLike (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/randomuniformlike.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~RandomUniformLike () {}
        

    };
}

#endif
