
#ifndef CONV_H
#define CONV_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Conv : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		std::string auto_pad;
		int[] dilations;
		int group;
		int[] kernel_shape;
		int[] pads;
		int[] strides;
    };
    vuh::Program<Specs, Params>* program;

    public:
       Conv (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/conv.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~Conv () {}

    };
}

#endif
