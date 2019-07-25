
#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class MaxRoiPool : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int[] pooled_shape;
		float spatial_scale;
    };
    vuh::Program<Specs, Params>* program;

    public:
       MaxRoiPool (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/maxroipool.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~MaxRoiPool () {}

    };
}

#endif
