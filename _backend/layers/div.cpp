#include "Div.h"

//cpp stuff
namespace backend {    
   
    Div::Div(std::string n) : Layer(n) { }
       
    vuh::Device* Div::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Div::init() {      
    
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
 
		binding.C_output = tensor_dict[C_output]->shape();
 

    }
    
    void Div::call(std::string A_input, std::string B_input, std::string C_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/div.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[C_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


