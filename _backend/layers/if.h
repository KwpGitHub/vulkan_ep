
#ifndef IF_H
#define IF_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class If : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		//graph else_branch;
		//graph then_branch;
    };
    vuh::Program<Specs, Params>* program;

    public:
       If (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/if.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~If () {}

    };
}

#endif
