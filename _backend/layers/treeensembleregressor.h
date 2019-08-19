#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

input: Input of shape [N,F]
output: N classes
*/

//TreeEnsembleRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
//OPTIONAL_PARAMETERS_TYPE: int, Tensor*, int, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int, Shape_t, Shape_t, Shape_t, Tensor*

//class stuff
namespace backend {   

    class TreeEnsembleRegressor : public Layer {
        typedef struct {
            int aggregate_function; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids;
			Shape_t base_values; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values; Shape_t target_weights;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int aggregate_function; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids; std::string base_values; std::string nodes_hitrates; std::string nodes_modes; std::string nodes_values; std::string target_weights;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TreeEnsembleRegressor();
    
        void forward() { program->run(); }
        
        void init( int _aggregate_function,  int _n_targets,  Shape_t _nodes_falsenodeids,  Shape_t _nodes_featureids,  Shape_t _nodes_missing_value_tracks_true,  Shape_t _nodes_nodeids,  Shape_t _nodes_treeids,  Shape_t _nodes_truenodeids,  int _post_transform,  Shape_t _target_ids,  Shape_t _target_nodeids,  Shape_t _target_treeids); 
        void bind(std::string _base_values, std::string _nodes_hitrates, std::string _nodes_modes, std::string _nodes_values, std::string _target_weights, std::string _X_input, std::string _Y_output); 

        ~TreeEnsembleRegressor() {}
    };

    
    void init_layer_TreeEnsembleRegressor(py::module& m) {
        // py::class_(m, "TreeEnsembleRegressor");
    }
    

}


#endif

