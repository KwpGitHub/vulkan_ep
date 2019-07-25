
#ifndef LPPOOL_H
#define LPPOOL_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpPool : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		std::string auto_pad;
		int[] kernel_shape;
		int p;
		int[] pads;
		int[] strides;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LpPool (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/lppool.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LpPool () {}

    };
}

#endif
