
#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class OneHotEncoder : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] cats_int64s;
		std::string[] cats_strings;
		int zeros;
    };
    vuh::Program<Specs, Params>* program;

    public:
       OneHotEncoder (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/onehotencoder.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~OneHotEncoder () {}
        

    };
}

#endif
