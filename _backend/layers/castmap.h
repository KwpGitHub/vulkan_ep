
#ifndef CASTMAP_H
#define CASTMAP_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class CastMap : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		std::string cast_to;
		std::string map_form;
		int max_map;
    };
    vuh::Program<Specs, Params>* program;

    public:
       CastMap (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/castmap.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~CastMap () {}

    };
}

#endif
