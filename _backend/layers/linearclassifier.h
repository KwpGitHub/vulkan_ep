
#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LinearClassifier : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		int[] classlabels_ints;
		std::string[] classlabels_strings;
		float[] coefficients;
		float[] intercepts;
		int multi_class;
		std::string post_transform;
    };
    vuh::Program<Specs, Params>* program;

    public:
       LinearClassifier (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/linearclassifier.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~LinearClassifier () {}
        

    };
}

#endif
