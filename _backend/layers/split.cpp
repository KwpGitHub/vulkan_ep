#include "Split.h"

//cpp stuff
namespace backend {    
   
    Split::Split(std::string n, int axis, Shape_t split) : Layer(n) { }
       
    vuh::Device* Split::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Split::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 

		binding.axis = axis;
  		binding.split = split;
 
    }
    
    void Split::call(std::string input_input){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


