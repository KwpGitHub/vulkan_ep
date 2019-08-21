#include "SVMClassifier.h"
//cpp stuff
namespace backend {    
   
    SVMClassifier::SVMClassifier(std::string name) : Layer(name) { }
       
    vuh::Device* SVMClassifier::_get_device() {
        
        return device;
    }
    
    void SVMClassifier::init( Shape_t _classlabels_ints,  int _kernel_type,  int _post_transform,  Shape_t _vectors_per_class) {      
		 classlabels_ints = _classlabels_ints; 
 		 kernel_type = _kernel_type; 
 		 post_transform = _post_transform; 
 		 vectors_per_class = _vectors_per_class; 
  
    }
    
    void SVMClassifier::bind(std::string _classlabels_strings, std::string _coefficients, std::string _kernel_params, std::string _prob_a, std::string _prob_b, std::string _rho, std::string _support_vectors, std::string _X_i, std::string _Y_o, std::string _Z_o){
        classlabels_strings = _classlabels_strings; coefficients = _coefficients; kernel_params = _kernel_params; prob_a = _prob_a; prob_b = _prob_b; rho = _rho; support_vectors = _support_vectors; X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
  		binding.Z_o = tensor_dict[Z_o]->shape();
 
		binding.classlabels_ints = classlabels_ints;
  		binding.kernel_type = kernel_type;
  		binding.post_transform = post_transform;
  		binding.vectors_per_class = vectors_per_class;
 
		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.kernel_params = tensor_dict[kernel_params]->shape();
  		binding.prob_a = tensor_dict[prob_a]->shape();
  		binding.prob_b = tensor_dict[prob_b]->shape();
  		binding.rho = tensor_dict[rho]->shape();
  		binding.support_vectors = tensor_dict[support_vectors]->shape();
 
        
    }
}

