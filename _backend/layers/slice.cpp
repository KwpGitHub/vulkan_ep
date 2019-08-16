#include "Slice.h"

//cpp stuff
namespace backend {    
   
    Slice::Slice(std::string n) : Layer(n) { }
       
    vuh::Device* Slice::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Slice::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.starts_input = tensor_dict[starts_input]->shape();
  		binding.ends_input = tensor_dict[ends_input]->shape();
  		binding.axes_input_opt = tensor_dict[axes_input_opt]->shape();
  		binding.steps_input_opt = tensor_dict[steps_input_opt]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Slice::call(std::string data_input, std::string starts_input, std::string ends_input, std::string axes_input_opt, std::string steps_input_opt, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/slice.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[starts_input]->data(), *tensor_dict[ends_input]->data(), *tensor_dict[axes_input_opt]->data(), *tensor_dict[steps_input_opt]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


