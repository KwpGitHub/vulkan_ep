#include "treeensembleclassifier.h"
//cpp stuff
namespace layers {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/treeensembleclassifier.spv");       
        dev = backend::device;
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
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TreeEnsembleClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Z_o]->data);
    }

    void TreeEnsembleClassifier::forward(){ 
        program->run();
    }

}

