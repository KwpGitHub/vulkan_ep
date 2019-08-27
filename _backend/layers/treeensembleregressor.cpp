#include "treeensembleregressor.h"
//cpp stuff
namespace layers {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/treeensembleregressor.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* TreeEnsembleRegressor::_get_device() {        
        return backend::device;
    }
    
    void TreeEnsembleRegressor::init( std::string _aggregate_function,  std::vector<float> _base_values,  int _n_targets,  std::vector<int> _nodes_falsenodeids,  std::vector<int> _nodes_featureids,  std::vector<float> _nodes_hitrates,  std::vector<int> _nodes_missing_value_tracks_true,  std::vector<std::string> _nodes_modes,  std::vector<int> _nodes_nodeids,  std::vector<int> _nodes_treeids,  std::vector<int> _nodes_truenodeids,  std::vector<float> _nodes_values,  std::string _post_transform,  std::vector<int> _target_ids,  std::vector<int> _target_nodeids,  std::vector<int> _target_treeids,  std::vector<float> _target_weights) {      
		 aggregate_function = _aggregate_function; 
 		 base_values = _base_values; 
 		 n_targets = _n_targets; 
 		 nodes_falsenodeids = _nodes_falsenodeids; 
 		 nodes_featureids = _nodes_featureids; 
 		 nodes_hitrates = _nodes_hitrates; 
 		 nodes_missing_value_tracks_true = _nodes_missing_value_tracks_true; 
 		 nodes_modes = _nodes_modes; 
 		 nodes_nodeids = _nodes_nodeids; 
 		 nodes_treeids = _nodes_treeids; 
 		 nodes_truenodeids = _nodes_truenodeids; 
 		 nodes_values = _nodes_values; 
 		 post_transform = _post_transform; 
 		 target_ids = _target_ids; 
 		 target_nodeids = _target_nodeids; 
 		 target_treeids = _target_treeids; 
 		 target_weights = _target_weights; 
  
    }
    
    void TreeEnsembleRegressor::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.aggregate_function = aggregate_function;
  		//binding.base_values = base_values;
  		//binding.n_targets = n_targets;
  		//binding.nodes_falsenodeids = nodes_falsenodeids;
  		//binding.nodes_featureids = nodes_featureids;
  		//binding.nodes_hitrates = nodes_hitrates;
  		//binding.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true;
  		//binding.nodes_modes = nodes_modes;
  		//binding.nodes_nodeids = nodes_nodeids;
  		//binding.nodes_treeids = nodes_treeids;
  		//binding.nodes_truenodeids = nodes_truenodeids;
  		//binding.nodes_values = nodes_values;
  		//binding.post_transform = post_transform;
  		//binding.target_ids = target_ids;
  		//binding.target_nodeids = target_nodeids;
  		//binding.target_treeids = target_treeids;
  		//binding.target_weights = target_weights;
         
    }

    void TreeEnsembleRegressor::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void TreeEnsembleRegressor::forward(){ 
        program->run();
    }

}

