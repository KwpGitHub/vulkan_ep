#include "TopK.h"
//cpp stuff
namespace backend {    
   
    TopK::TopK(const std::string& name) : Layer(name) { }
       
    vuh::Device* TopK::_get_device() {
        
        return device;
    }
    
    void TopK::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void TopK::bind(std::string _X_input, std::string _K_input, std::string _Values_output, std::string _Indices_output){
        X_input = _X_input; K_input = _K_input; Values_output = _Values_output; Indices_output = _Indices_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.K_input = tensor_dict[K_input]->shape();
 
		binding.Values_output = tensor_dict[Values_output]->shape();
  		binding.Indices_output = tensor_dict[Indices_output]->shape();
 
		binding.axis = axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/topk.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[K_input]->data(), *tensor_dict[Values_output]->data(), *tensor_dict[Indices_output]->data());
    }

}

