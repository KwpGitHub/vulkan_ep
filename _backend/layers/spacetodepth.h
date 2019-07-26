
#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class SpaceToDepth : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int blocksize;
    };
    vuh::Program<Specs, Params>* program;

    public:
       SpaceToDepth (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/spacetodepth.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~SpaceToDepth () {}
        

    };
}

#endif
