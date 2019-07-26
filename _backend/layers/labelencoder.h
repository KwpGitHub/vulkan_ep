
#ifndef LABELENCODER_H
#define LABELENCODER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LabelEncoder : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float default_float;
		int default_int64;
		std::string default_string;
		float[] keys_floats;
		int[] keys_int64s;
		std::string[] keys_strings;
		float[] values_floats;
		int[] values_int64s;
		std::string[] values_strings;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LabelEncoder (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/labelencoder.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LabelEncoder () {}
        

    };
}

#endif
