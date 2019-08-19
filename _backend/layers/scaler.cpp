#include "Scaler.h"
//cpp stuff
namespace backend {    
   
    Scaler::Scaler() : Layer() { }
       
    vuh::Device* Scaler::_get_device() {
        
        return device;
    }
    
    void Scaler::init() {      
  
    }
    
    void Scaler::bind(std::string _offset, std::string _scale, std::string _X_input, std::string _Y_output){
        offset = _offset; scale = _scale; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 

		binding.offset = tensor_dict[offset]->shape();
  		binding.scale = tensor_dict[scale]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scaler.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[offset]->data(), *tensor_dict[scale]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



