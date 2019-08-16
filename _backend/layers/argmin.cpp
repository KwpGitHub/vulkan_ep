#include "ArgMin.h"

//cpp stuff
namespace backend {    
   
    ArgMin::ArgMin(std::string n, int axis, int keepdims) : Layer(n) { }
       
    vuh::Device* ArgMin::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArgMin::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.reduced_output = tensor_dict[reduced_output]->shape();
 
		binding.axis = axis;
  		binding.keepdims = keepdims;
 
    }
    
    void ArgMin::call(std::string data_input, std::string reduced_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmin.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[reduced_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


