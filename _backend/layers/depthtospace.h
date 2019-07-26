
#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class DepthToSpace : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int blocksize;
    };
    vuh::Program<Specs, Params>* program;

    public:
       DepthToSpace (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/depthtospace.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~DepthToSpace () {}
        

    };
}

#endif
