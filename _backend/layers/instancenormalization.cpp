#include "InstanceNormalization.h"

//cpp stuff
namespace backend {    
   
    InstanceNormalization::InstanceNormalization(std::string n, float epsilon) : Layer(n) { }
       
    vuh::Device* InstanceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void InstanceNormalization::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.scale_input = tensor_dict[scale_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.epsilon = epsilon;
 
    }
    
    void InstanceNormalization::call(std::string input_input, std::string scale_input, std::string B_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/instancenormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[scale_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


