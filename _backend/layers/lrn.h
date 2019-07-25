
#ifndef LRN_H
#define LRN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LRN : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		float alpha;
		float beta;
		float bias;
		int size;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LRN (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/lrn.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LRN () {}

    };
}

#endif
