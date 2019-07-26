
#ifndef GLOBALAVERAGEPOOL_H
#define GLOBALAVERAGEPOOL_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalAveragePool : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
    };
    vuh::Program<Specs, Params>* program;

    public:
       GlobalAveragePool (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/globalaveragepool.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~GlobalAveragePool () {}
        

    };
}

#endif
