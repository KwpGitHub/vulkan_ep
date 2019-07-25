
#ifndef RNN_H
#define RNN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RNN : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		float[] activation_alpha;
		float[] activation_beta;
		std::string[] activations;
		float clip;
		std::string direction;
		int hidden_size;
    };
    vuh::Program<Specs, Params>* program;

    public:
       RNN (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/rnn.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~RNN () {}

    };
}

#endif
