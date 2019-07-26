
#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMClassifier : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] classlabels_ints;
		std::string[] classlabels_strings;
		float[] coefficients;
		float[] kernel_params;
		std::string kernel_type;
		std::string post_transform;
		float[] prob_a;
		float[] prob_b;
		float[] rho;
		float[] support_vectors;
		int[] vectors_per_class;
    };
    vuh::Program<Specs, Params>* program;

    public:
       SVMClassifier (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/svmclassifier.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~SVMClassifier () {}
        

    };
}

#endif
