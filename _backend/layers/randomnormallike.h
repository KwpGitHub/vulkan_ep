
#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormalLike : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int dtype;
		float mean;
		float scale;
		float seed;
    };
    vuh::Program<Specs, Params>* program;

    public:
       RandomNormalLike (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/randomnormallike.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~RandomNormalLike () {}
        

    };
}

#endif
