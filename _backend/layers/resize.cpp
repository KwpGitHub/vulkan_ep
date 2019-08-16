#include "Resize.h"

//cpp stuff
namespace backend {    
   
    Resize::Resize(std::string n, int mode) : Layer(n) { }
       
    vuh::Device* Resize::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Resize::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.scales_input = tensor_dict[scales_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.mode = mode;
 
    }
    
    void Resize::call(std::string X_input, std::string scales_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/resize.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[scales_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


