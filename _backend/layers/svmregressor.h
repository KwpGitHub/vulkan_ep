
#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMRegressor : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float[] coefficients;
		float[] kernel_params;
		std::string kernel_type;
		int n_supports;
		int one_class;
		std::string post_transform;
		float[] rho;
		float[] support_vectors;
    };
    vuh::Program<Specs, Params>* program;

    public:
       SVMRegressor (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/svmregressor.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~SVMRegressor () {}
        

    };
}

#endif
