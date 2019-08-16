#include "Gather.h"

//cpp stuff
namespace backend {    
   
    Gather::Gather(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* Gather::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Gather::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.indices_input = tensor_dict[indices_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void Gather::call(std::string data_input, std::string indices_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gather.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[indices_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


