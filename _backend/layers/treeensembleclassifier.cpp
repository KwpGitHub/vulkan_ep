#include "treeensembleclassifier.h"
//cpp stuff
namespace layers {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/treeensembleclassifier.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* TreeEnsembleClassifier::_get_device() {
        
        return backend::device;
    }
    
    void TreeEnsembleClassifier::init( std::vector<float> _base_values,  std::vector<int> _class_ids,  std::vector<int> _class_nodeids,  std::vector<int> _class_treeids,  std::vector<float> _class_weights,  std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings,  std::vector<int> _nodes_falsenodeids,  std::vector<int> _nodes_featureids,  std::vector<float> _nodes_hitrates,  std::vector<int> _nodes_missing_value_tracks_true,  std::vector<std::string> _nodes_modes,  std::vector<int> _nodes_nodeids,  std::vector<int> _nodes_treeids,  std::vector<int> _nodes_truenodeids,  std::vector<float> _nodes_values,  std::string _post_transform) {      
		 base_values = _base_values; 
 		 class_ids = _class_ids; 
 		 class_nodeids = _class_nodeids; 
 		 class_treeids = _class_treeids; 
 		 class_weights = _class_weights; 
 		 classlabels_int64s = _classlabels_int64s; 
 		 classlabels_strings = _classlabels_strings; 
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
  
    }
    
    void TreeEnsembleClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){
        X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
  		//binding.Z_o = tensor_dict[Z_o]->shape();
 
		//binding.base_values = base_values;
  		//binding.class_ids = class_ids;
  		//binding.class_nodeids = class_nodeids;
  		//binding.class_treeids = class_treeids;
  		//binding.class_weights = class_weights;
  		//binding.classlabels_int64s = classlabels_int64s;
  		//binding.classlabels_strings = classlabels_strings;
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
         
    }

    void TreeEnsembleClassifier::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[Z_o]->data());
    }

}

