
#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleClassifier : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {
		float[] base_values;
		int[] class_ids;
		int[] class_nodeids;
		int[] class_treeids;
		float[] class_weights;
		int[] classlabels_int64s;
		std::string[] classlabels_strings;
		int[] nodes_falsenodeids;
		int[] nodes_featureids;
		float[] nodes_hitrates;
		int[] nodes_missing_value_tracks_true;
		std::string[] nodes_modes;
		int[] nodes_nodeids;
		int[] nodes_treeids;
		int[] nodes_truenodeids;
		float[] nodes_values;
		std::string post_transform;
    };
    vuh::Program<Specs, Params>* program;

    public:
       TreeEnsembleClassifier (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/treeensembleclassifier.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~TreeEnsembleClassifier () {}
        

    };
}

#endif
