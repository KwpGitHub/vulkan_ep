#include "TreeEnsembleClassifier.h"
//cpp stuff
namespace backend {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier() : Layer() { }
       
    vuh::Device* TreeEnsembleClassifier::_get_device() {
        
        return device;
    }
    
    void TreeEnsembleClassifier::init( Shape_t _class_ids,  Shape_t _class_nodeids,  Shape_t _class_treeids,  Shape_t _classlabels_int64s,  Shape_t _nodes_falsenodeids,  Shape_t _nodes_featureids,  Shape_t _nodes_missing_value_tracks_true,  Shape_t _nodes_nodeids,  Shape_t _nodes_treeids,  Shape_t _nodes_truenodeids,  int _post_transform) {      
		 class_ids = _class_ids; 
 		 class_nodeids = _class_nodeids; 
 		 class_treeids = _class_treeids; 
 		 classlabels_int64s = _classlabels_int64s; 
 		 nodes_falsenodeids = _nodes_falsenodeids; 
 		 nodes_featureids = _nodes_featureids; 
 		 nodes_missing_value_tracks_true = _nodes_missing_value_tracks_true; 
 		 nodes_nodeids = _nodes_nodeids; 
 		 nodes_treeids = _nodes_treeids; 
 		 nodes_truenodeids = _nodes_truenodeids; 
 		 post_transform = _post_transform; 
  
    }
    
    void TreeEnsembleClassifier::bind(std::string _base_values, std::string _class_weights, std::string _classlabels_strings, std::string _nodes_hitrates, std::string _nodes_modes, std::string _nodes_values, std::string _X_input, std::string _Y_output, std::string _Z_output){
        base_values = _base_values; class_weights = _class_weights; classlabels_strings = _classlabels_strings; nodes_hitrates = _nodes_hitrates; nodes_modes = _nodes_modes; nodes_values = _nodes_values; X_input = _X_input; Y_output = _Y_output; Z_output = _Z_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.class_ids = class_ids;
  		binding.class_nodeids = class_nodeids;
  		binding.class_treeids = class_treeids;
  		binding.classlabels_int64s = classlabels_int64s;
  		binding.nodes_falsenodeids = nodes_falsenodeids;
  		binding.nodes_featureids = nodes_featureids;
  		binding.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true;
  		binding.nodes_nodeids = nodes_nodeids;
  		binding.nodes_treeids = nodes_treeids;
  		binding.nodes_truenodeids = nodes_truenodeids;
  		binding.post_transform = post_transform;
 
		binding.base_values = tensor_dict[base_values]->shape();
  		binding.class_weights = tensor_dict[class_weights]->shape();
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.nodes_hitrates = tensor_dict[nodes_hitrates]->shape();
  		binding.nodes_modes = tensor_dict[nodes_modes]->shape();
  		binding.nodes_values = tensor_dict[nodes_values]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleclassifier.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[base_values]->data(), *tensor_dict[class_weights]->data(), *tensor_dict[classlabels_strings]->data(), *tensor_dict[nodes_hitrates]->data(), *tensor_dict[nodes_modes]->data(), *tensor_dict[nodes_values]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Z_output]->data());
    }



}



