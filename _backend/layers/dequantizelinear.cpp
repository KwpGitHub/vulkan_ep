#include "DequantizeLinear.h"

//cpp stuff
namespace backend {    
   
    DequantizeLinear::DequantizeLinear(std::string n) : Layer(n) { }
       
    vuh::Device* DequantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void DequantizeLinear::init() {      
    
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.x_scale_input = tensor_dict[x_scale_input]->shape();
  		binding.x_zero_point_input_opt = tensor_dict[x_zero_point_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 

    }
    
    void DequantizeLinear::call(std::string x_input, std::string x_scale_input, std::string x_zero_point_input_opt, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dequantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[x_scale_input]->data(), *tensor_dict[x_zero_point_input_opt]->data(), *tensor_dict[y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


