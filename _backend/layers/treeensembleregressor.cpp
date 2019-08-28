#include "treeensembleregressor.h"
//cpp stuff
namespace layers {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/treeensembleregressor.spv");       
        dev = backend::device;
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
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TreeEnsembleRegressor::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void TreeEnsembleRegressor::forward(){ 
        program->run();
    }

}

