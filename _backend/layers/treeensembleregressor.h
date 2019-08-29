#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H 

#include "../layer.h"

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
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
//OPTIONAL_PARAMETERS_TYPE: std::string, std::vector<float>, int, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<int>, std::vector<std::string>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<float>, std::string, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<float>


//class stuff
namespace layers {   

    class TreeEnsembleRegressor : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string aggregate_function; std::vector<float> base_values; int n_targets; std::vector<int> nodes_falsenodeids; std::vector<int> nodes_featureids; std::vector<float> nodes_hitrates; std::vector<int> nodes_missing_value_tracks_true; std::vector<std::string> nodes_modes; std::vector<int> nodes_nodeids; std::vector<int> nodes_treeids; std::vector<int> nodes_truenodeids; std::vector<float> nodes_values; std::string post_transform; std::vector<int> target_ids; std::vector<int> target_nodeids; std::vector<int> target_treeids; std::vector<float> target_weights;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        TreeEnsembleRegressor(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _aggregate_function,  std::vector<float> _base_values,  int _n_targets,  std::vector<int> _nodes_falsenodeids,  std::vector<int> _nodes_featureids,  std::vector<float> _nodes_hitrates,  std::vector<int> _nodes_missing_value_tracks_true,  std::vector<std::string> _nodes_modes,  std::vector<int> _nodes_nodeids,  std::vector<int> _nodes_treeids,  std::vector<int> _nodes_truenodeids,  std::vector<float> _nodes_values,  std::string _post_transform,  std::vector<int> _target_ids,  std::vector<int> _target_nodeids,  std::vector<int> _target_treeids,  std::vector<float> _target_weights); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~TreeEnsembleRegressor() {}
    };
   
}
#endif

