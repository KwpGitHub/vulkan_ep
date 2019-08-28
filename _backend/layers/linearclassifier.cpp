#include "linearclassifier.h"
//cpp stuff
namespace layers {    
   
    LinearClassifier::LinearClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/linearclassifier.spv");       
        dev = backend::device;
    }
       
        
    void LinearClassifier::init( std::vector<float> _coefficients,  std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _intercepts,  int _multi_class,  std::string _post_transform) {      
		 coefficients = _coefficients; 
 		 classlabels_ints = _classlabels_ints; 
 		 classlabels_strings = _classlabels_strings; 
 		 intercepts = _intercepts; 
 		 multi_class = _multi_class; 
 		 post_transform = _post_transform; 
  

    }
    
    void LinearClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){    
        X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LinearClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Z_o]->data);
    }

    void LinearClassifier::forward(){ 
        program->run();
    }

}

