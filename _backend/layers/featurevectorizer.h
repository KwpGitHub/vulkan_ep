
#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class FeatureVectorizer : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		int[] inputdimensions;
    };
    vuh::Program<Specs, Params>* program;

    public:
       FeatureVectorizer (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/featurevectorizer.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~FeatureVectorizer () {}

    };
}

#endif
