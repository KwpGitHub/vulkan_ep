#include "LinearClassifier.h"
//cpp stuff
namespace backend {    
   
    LinearClassifier::LinearClassifier(const std::string& name) : Layer(name) { }
       
    vuh::Device* LinearClassifier::_get_device() {
        
        return device;
    }
    
    void LinearClassifier::init( Shape_t _classlabels_ints,  int _multi_class,  int _post_transform) {      
		 classlabels_ints = _classlabels_ints; 
 		 multi_class = _multi_class; 
 		 post_transform = _post_transform; 
  
    }
    
    void LinearClassifier::bind(std::string _coefficients, std::string _classlabels_strings, std::string _intercepts, std::string _X_i, std::string _Y_o, std::string _Z_o){
        coefficients = _coefficients; classlabels_strings = _classlabels_strings; intercepts = _intercepts; X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
  		binding.Z_o = tensor_dict[Z_o]->shape();
 
		binding.classlabels_ints = classlabels_ints;
  		binding.multi_class = multi_class;
  		binding.post_transform = post_transform;
 
		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearclassifier.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[classlabels_strings]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[Z_o]->data());
    }

}

