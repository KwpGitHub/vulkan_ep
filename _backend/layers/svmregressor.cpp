#include "SVMRegressor.h"

//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string n, int kernel_type, int n_supports, int one_class, int post_transform) : Layer(n) { }
       
    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMRegressor::init() {      
    
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
 
    }
    
    void SVMRegressor::call(std::string coefficients, std::string kernel_params, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[kernel_params]->data(), *tensor_dict[rho]->data(), *tensor_dict[support_vectors]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


