#include "TreeEnsembleRegressor.h"
//cpp stuff
namespace backend {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(const std::string& name) : Layer(name) { }
       
    vuh::Device* TreeEnsembleRegressor::_get_device() {
        
        return device;
    }
    
    void TreeEnsembleRegressor::init( int _aggregate_function,  int _n_targets,  Shape_t _nodes_falsenodeids,  Shape_t _nodes_featureids,  Shape_t _nodes_missing_value_tracks_true,  Shape_t _nodes_nodeids,  Shape_t _nodes_treeids,  Shape_t _nodes_truenodeids,  int _post_transform,  Shape_t _target_ids,  Shape_t _target_nodeids,  Shape_t _target_treeids) {      
		 aggregate_function = _aggregate_function; 
 		 n_targets = _n_targets; 
 		 nodes_falsenodeids = _nodes_falsenodeids; 
 		 nodes_featureids = _nodes_featureids; 
 		 nodes_missing_value_tracks_true = _nodes_missing_value_tracks_true; 
 		 nodes_nodeids = _nodes_nodeids; 
 		 nodes_treeids = _nodes_treeids; 
 		 nodes_truenodeids = _nodes_truenodeids; 
 		 post_transform = _post_transform; 
 		 target_ids = _target_ids; 
 		 target_nodeids = _target_nodeids; 
 		 target_treeids = _target_treeids; 
  
    }
    
    void TreeEnsembleRegressor::bind(std::string _base_values, std::string _nodes_hitrates, std::string _nodes_modes, std::string _nodes_values, std::string _target_weights, std::string _X_i, std::string _Y_o){
        base_values = _base_values; nodes_hitrates = _nodes_hitrates; nodes_modes = _nodes_modes; nodes_values = _nodes_values; target_weights = _target_weights; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
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
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleregressor.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[base_values]->data(), *tensor_dict[nodes_hitrates]->data(), *tensor_dict[nodes_modes]->data(), *tensor_dict[nodes_values]->data(), *tensor_dict[target_weights]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

