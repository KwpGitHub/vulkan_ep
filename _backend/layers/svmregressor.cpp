#include "SVMRegressor.h"
//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string name) : Layer(name) { }
       
    vuh::Device* SVMRegressor::_get_device() {
        
        return device;
    }
    
    void SVMRegressor::init( int _kernel_type,  int _n_supports,  int _one_class,  int _post_transform) {      
		 kernel_type = _kernel_type; 
 		 n_supports = _n_supports; 
 		 one_class = _one_class; 
 		 post_transform = _post_transform; 
  
    }
    
    void SVMRegressor::bind(std::string _coefficients, std::string _kernel_params, std::string _rho, std::string _support_vectors, std::string _X_i, std::string _Y_o){
        coefficients = _coefficients; kernel_params = _kernel_params; rho = _rho; support_vectors = _support_vectors; X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.kernel_type = kernel_type;
  		binding.n_supports = n_supports;
  		binding.one_class = one_class;
  		binding.post_transform = post_transform;
 
		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.kernel_params = tensor_dict[kernel_params]->shape();
  		binding.rho = tensor_dict[rho]->shape();
  		binding.support_vectors = tensor_dict[support_vectors]->shape();
 
        
    }
}

