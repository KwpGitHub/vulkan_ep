
#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleRegressor : public Layer {
    using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
    struct Params {

		std::string aggregate_function;
		float[] base_values;
		int n_targets;
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
		int[] target_ids;
		int[] target_nodeids;
		int[] target_treeids;
		float[] target_weights;
    };
    vuh::Program<Specs, Params>* program;

    public:
       TreeEnsembleRegressor (){
            device =  new vuh::Device(instance->devices().at(0));
		    program = new vuh::Program<Specs, Params>(*device, "../shaders/bin/treeensembleregressor.spv");
		    d_input = new vuh::Array<float>(*device, input);
		    d_output = new vuh::Array<float>(*device, output);
        }

        ~TreeEnsembleRegressor () {}

    };
}

#endif
