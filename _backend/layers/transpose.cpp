#include "Transpose.h"

//cpp stuff
namespace backend {    
   
    Transpose::Transpose(std::string n, Shape_t perm) : Layer(n) { }
       
    vuh::Device* Transpose::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Transpose::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.transposed_output = tensor_dict[transposed_output]->shape();
 
		binding.perm = perm;
 
    }
    
    void Transpose::call(std::string data_input, std::string transposed_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/transpose.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[transposed_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


