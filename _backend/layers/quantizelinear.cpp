#include "QuantizeLinear.h"

//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(std::string n) : Layer(n) { }
       
    vuh::Device* QuantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QuantizeLinear::init() {      
    
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input_opt = tensor_dict[y_zero_point_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 

    }
    
    void QuantizeLinear::call(std::string x_input, std::string y_scale_input, std::string y_zero_point_input_opt, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input_opt]->data(), *tensor_dict[y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


