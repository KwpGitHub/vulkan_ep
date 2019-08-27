#include "linearclassifier.h"
//cpp stuff
namespace layers {    
   
    LinearClassifier::LinearClassifier(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/linearclassifier.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* LinearClassifier::_get_device() {        
        return backend::device;
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

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
  		binding.Z_o = backend::tensor_dict[Z_o]->shape();
 
		//binding.coefficients = coefficients;
  		//binding.classlabels_ints = classlabels_ints;
  		//binding.classlabels_strings = classlabels_strings;
  		//binding.intercepts = intercepts;
  		//binding.multi_class = multi_class;
  		//binding.post_transform = post_transform;
         
    }

    void LinearClassifier::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data(), *backend::tensor_dict[Z_o]->data());
    }

    void LinearClassifier::forward(){ 
        //program->run();
    }

}

