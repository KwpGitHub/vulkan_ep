#include "TreeEnsembleRegressor.h"

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

    py::module m("_backend.nn", "nn MOD");

//python stuff


