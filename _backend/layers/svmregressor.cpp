#include "svmregressor.h"
//cpp stuff
namespace layers {    
   
    SVMRegressor::SVMRegressor(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/svmregressor.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* SVMRegressor::_get_device() {        
        return backend::device;
    }
    
    void SVMRegressor::init( std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  int _n_supports,  int _one_class,  std::string _post_transform,  std::vector<float> _rho,  std::vector<float> _support_vectors) {      
		 coefficients = _coefficients; 
 		 kernel_params = _kernel_params; 
 		 kernel_type = _kernel_type; 
 		 n_supports = _n_supports; 
 		 one_class = _one_class; 
 		 post_transform = _post_transform; 
 		 rho = _rho; 
 		 support_vectors = _support_vectors; 
  
    }
    
    void SVMRegressor::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.coefficients = coefficients;
  		//binding.kernel_params = kernel_params;
  		//binding.kernel_type = kernel_type;
  		//binding.n_supports = n_supports;
  		//binding.one_class = one_class;
  		//binding.post_transform = post_transform;
  		//binding.rho = rho;
  		//binding.support_vectors = support_vectors;
         
    }

    void SVMRegressor::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void SVMRegressor::forward(){ 
        program->run();
    }

}

