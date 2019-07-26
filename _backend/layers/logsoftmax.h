
#ifndef LOGSOFTMAX_H
#define LOGSOFTMAX_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LogSoftmax : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int axis;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LogSoftmax (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/logsoftmax.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LogSoftmax () {}
        

    };
}

#endif
