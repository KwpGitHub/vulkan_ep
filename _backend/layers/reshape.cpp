#include "Reshape.h"

//cpp stuff
namespace backend {    
   
    Reshape::Reshape(std::string n) : Layer(n) { }
       
    vuh::Device* Reshape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Reshape::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.shape_input = tensor_dict[shape_input]->shape();
 
		binding.reshaped_output = tensor_dict[reshaped_output]->shape();
 

    }
    
    void Reshape::call(std::string data_input, std::string shape_input, std::string reshaped_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reshape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[shape_input]->data(), *tensor_dict[reshaped_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


