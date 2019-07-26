
#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class CategoryMapper : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] cats_int64s;
		std::string[] cats_strings;
		int default_int64;
		std::string default_string;
    };
    vuh::Program<Specs, Params>* program;

    public:
       CategoryMapper (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/categorymapper.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~CategoryMapper () {}
        

    };
}

#endif
