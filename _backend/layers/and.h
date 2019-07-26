
#ifndef AND_H
#define AND_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class And : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
    };
    vuh::Program<Specs, Params>* program;

    public:
       And (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/and.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~And () {}
        

    };
}

#endif
