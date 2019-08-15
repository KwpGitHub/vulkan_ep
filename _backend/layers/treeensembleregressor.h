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

namespace py = pybind11;

//class stuff
namespace backend {   

    class TreeEnsembleRegressor : public Layer {
        typedef struct {    
            int aggregate_function; Tensor* base_values; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Tensor* nodes_hitrates; Shape_t nodes_missing_value_tracks_true; Tensor* nodes_modes; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; Tensor* nodes_values; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids; Tensor* target_weights;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int aggregate_function; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids;
		Shape_t base_values; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values; Shape_t target_weights;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TreeEnsembleRegressor(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~TreeEnsembleRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* TreeEnsembleRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TreeEnsembleRegressor::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.aggregate_function = parameters.aggregate_function;
  		binding.n_targets = parameters.n_targets;
  		binding.nodes_falsenodeids = parameters.nodes_falsenodeids;
  		binding.nodes_featureids = parameters.nodes_featureids;
  		binding.nodes_missing_value_tracks_true = parameters.nodes_missing_value_tracks_true;
  		binding.nodes_nodeids = parameters.nodes_nodeids;
  		binding.nodes_treeids = parameters.nodes_treeids;
  		binding.nodes_truenodeids = parameters.nodes_truenodeids;
  		binding.post_transform = parameters.post_transform;
  		binding.target_ids = parameters.target_ids;
  		binding.target_nodeids = parameters.target_nodeids;
  		binding.target_treeids = parameters.target_treeids;
  		binding.base_values = parameters.base_values->shape();
  		binding.nodes_hitrates = parameters.nodes_hitrates->shape();
  		binding.nodes_modes = parameters.nodes_modes->shape();
  		binding.nodes_values = parameters.nodes_values->shape();
  		binding.target_weights = parameters.target_weights->shape();
 
        program->bind(binding, *parameters.base_values->data(), *parameters.nodes_hitrates->data(), *parameters.nodes_modes->data(), *parameters.nodes_values->data(), *parameters.target_weights->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void TreeEnsembleRegressor::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleRegressor, Layer>(m, "TreeEnsembleRegressor")
            .def("forward", &TreeEnsembleRegressor::forward);    
    }
}*/

#endif
