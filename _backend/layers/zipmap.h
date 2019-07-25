
#ifndef ZIPMAP_H
#define ZIPMAP_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ZipMap : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int[] classlabels_int64s;
		std::string[] classlabels_strings;
    };
    vuh::Program<Specs, Params>* program;

    public:
       ZipMap (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/zipmap.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~ZipMap () {}

    };
}

#endif
