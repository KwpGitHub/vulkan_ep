#include "Resize.h"
//cpp stuff
namespace backend {    
   
    Resize::Resize(const std::string& name) : Layer(name) { }
       
    vuh::Device* Resize::_get_device() {
        
        return device;
    }
    
    void Resize::init( int _mode) {      
		 mode = _mode; 
  
    }
    
    void Resize::bind(std::string _X_input, std::string _scales_input, std::string _Y_output){
        X_input = _X_input; scales_input = _scales_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.scales_input = tensor_dict[scales_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.mode = mode;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/resize.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[scales_input]->data(), *tensor_dict[Y_output]->data());
    }

}

