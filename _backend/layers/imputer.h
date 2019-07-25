
#ifndef IMPUTER_H
#define IMPUTER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Imputer : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		float[] imputed_value_floats;
		int[] imputed_value_int64s;
		float replaced_value_float;
		int replaced_value_int64;
    };
    vuh::Program<Specs, Params>* program;

    public:
       Imputer (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/imputer.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~Imputer () {}

    };
}

#endif
