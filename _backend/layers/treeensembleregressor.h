#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H 
#include <pybind11/pybind11.h>
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
//*/
//TreeEnsembleRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
//OPTIONAL_PARAMETERS_TYPE: int, Tensor*, int, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int, Shape_t, Shape_t, Shape_t, Tensor*

namespace py = pybind11;

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
        TreeEnsembleRegressor(std::string n, int aggregate_function, int n_targets, Shape_t nodes_falsenodeids, Shape_t nodes_featureids, Shape_t nodes_missing_value_tracks_true, Shape_t nodes_nodeids, Shape_t nodes_treeids, Shape_t nodes_truenodeids, int post_transform, Shape_t target_ids, Shape_t target_nodeids, Shape_t target_treeids);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string base_values, std::string nodes_hitrates, std::string nodes_modes, std::string nodes_values, std::string target_weights, std::string X_input, std::string Y_output); 

        ~TreeEnsembleRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string n, int aggregate_function, int n_targets, Shape_t nodes_falsenodeids, Shape_t nodes_featureids, Shape_t nodes_missing_value_tracks_true, Shape_t nodes_nodeids, Shape_t nodes_treeids, Shape_t nodes_truenodeids, int post_transform, Shape_t target_ids, Shape_t target_nodeids, Shape_t target_treeids) : Layer(n) { }
       
    vuh::Device* TreeEnsembleRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TreeEnsembleRegressor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.aggregate_function = aggregate_function;
  		binding.n_targets = n_targets;
  		binding.nodes_falsenodeids = nodes_falsenodeids;
  		binding.nodes_featureids = nodes_featureids;
  		binding.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true;
  		binding.nodes_nodeids = nodes_nodeids;
  		binding.nodes_treeids = nodes_treeids;
  		binding.nodes_truenodeids = nodes_truenodeids;
  		binding.post_transform = post_transform;
  		binding.target_ids = target_ids;
  		binding.target_nodeids = target_nodeids;
  		binding.target_treeids = target_treeids;
  		binding.base_values = tensor_dict[base_values]->shape();
  		binding.nodes_hitrates = tensor_dict[nodes_hitrates]->shape();
  		binding.nodes_modes = tensor_dict[nodes_modes]->shape();
  		binding.nodes_values = tensor_dict[nodes_values]->shape();
  		binding.target_weights = tensor_dict[target_weights]->shape();
 
    }
    
    void TreeEnsembleRegressor::call(std::string base_values, std::string nodes_hitrates, std::string nodes_modes, std::string nodes_values, std::string target_weights, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[base_values]->data(), *tensor_dict[nodes_hitrates]->data(), *tensor_dict[nodes_modes]->data(), *tensor_dict[nodes_values]->data(), *tensor_dict[target_weights]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleRegressor, Layer>(m, "TreeEnsembleRegressor")
            .def(py::init<std::string, int, int, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, int, Shape_t, Shape_t, Shape_t> ())
            .def("forward", &TreeEnsembleRegressor::forward)
            .def("init", &TreeEnsembleRegressor::init)
            .def("call", (void (TreeEnsembleRegressor::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &TreeEnsembleRegressor::call);
    }
}

#endif

/* PYTHON STUFF

*/

