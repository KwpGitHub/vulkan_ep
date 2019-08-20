#include "Scaler.h"
//cpp stuff
namespace backend {    
   
    Scaler::Scaler(const std::string& name) : Layer(name) { }
       
    vuh::Device* Scaler::_get_device() {
        
        return device;
    }
    
    void Scaler::init() {      
  
    }
    
    void Scaler::bind(std::string _offset, std::string _scale, std::string _X_i, std::string _Y_o){
        offset = _offset; scale = _scale; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 

		binding.offset = tensor_dict[offset]->shape();
  		binding.scale = tensor_dict[scale]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scaler.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[offset]->data(), *tensor_dict[scale]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

