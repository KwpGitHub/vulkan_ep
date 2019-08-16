#include "OneHot.h"

//cpp stuff
namespace backend {    
   
    OneHot::OneHot(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* OneHot::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void OneHot::init() {      
    
		binding.indices_input = tensor_dict[indices_input]->shape();
  		binding.depth_input = tensor_dict[depth_input]->shape();
  		binding.values_input = tensor_dict[values_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void OneHot::call(std::string indices_input, std::string depth_input, std::string values_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehot.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[indices_input]->data(), *tensor_dict[depth_input]->data(), *tensor_dict[values_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


