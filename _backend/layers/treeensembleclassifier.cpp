#include "treeensembleclassifier.h"
//cpp stuff
namespace layers {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/treeensembleclassifier.spv");       
        dev = backend::g_device;
    }
       
        
    void TreeEnsembleClassifier::init( std::vector<float> _base_values,  std::vector<int> _class_ids,  std::vector<int> _class_nodeids,  std::vector<int> _class_treeids,  std::vector<float> _class_weights,  std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings,  std::vector<int> _nodes_falsenodeids,  std::vector<int> _nodes_featureids,  std::vector<float> _nodes_hitrates,  std::vector<int> _nodes_missing_value_tracks_true,  std::vector<std::string> _nodes_modes,  std::vector<int> _nodes_nodeids,  std::vector<int> _nodes_treeids,  std::vector<int> _nodes_truenodeids,  std::vector<float> _nodes_values,  std::string _post_transform) {      
		 m_base_values = _base_values; 
 		 m_class_ids = _class_ids; 
 		 m_class_nodeids = _class_nodeids; 
 		 m_class_treeids = _class_treeids; 
 		 m_class_weights = _class_weights; 
 		 m_classlabels_int64s = _classlabels_int64s; 
 		 m_classlabels_strings = _classlabels_strings; 
 		 m_nodes_falsenodeids = _nodes_falsenodeids; 
 		 m_nodes_featureids = _nodes_featureids; 
 		 m_nodes_hitrates = _nodes_hitrates; 
 		 m_nodes_missing_value_tracks_true = _nodes_missing_value_tracks_true; 
 		 m_nodes_modes = _nodes_modes; 
 		 m_nodes_nodeids = _nodes_nodeids; 
 		 m_nodes_treeids = _nodes_treeids; 
 		 m_nodes_truenodeids = _nodes_truenodeids; 
 		 m_nodes_values = _nodes_values; 
 		 m_post_transform = _post_transform; 
  

    }
    
    void TreeEnsembleClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){    
        m_X_i = _X_i; m_Y_o = _Y_o; m_Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TreeEnsembleClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Z_o]->data);       
    }

    void TreeEnsembleClassifier::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Z_o]->data);
        //program->run();
    }

}

