#include "SVMRegressor.h"

//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string n) : Layer(n) { }
       
    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMRegressor::init( int _kernel_type,  int _n_supports,  int _one_class,  int _post_transform) {      
		 kernel_type = _kernel_type; 
 		 n_supports = _n_supports; 
 		 one_class = _one_class; 
 		 post_transform = _post_transform; 
  
    }
    
    void SVMRegressor::bind(std::string _coefficients, std::string _kernel_params, std::string _rho, std::string _support_vectors, std::string _X_input, std::string _Y_output){
        coefficients = _coefficients; kernel_params = _kernel_params; rho = _rho; support_vectors = _support_vectors; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.kernel_type = kernel_type;
  		binding.n_supports = n_supports;
  		binding.one_class = one_class;
  		binding.post_transform = post_transform;
 
		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.kernel_params = tensor_dict[kernel_params]->shape();
  		binding.rho = tensor_dict[rho]->shape();
  		binding.support_vectors = tensor_dict[support_vectors]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmregressor.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[kernel_params]->data(), *tensor_dict[rho]->data(), *tensor_dict[support_vectors]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


