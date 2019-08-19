#include "QuantizeLinear.h"
//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear() : Layer() { }
       
    vuh::Device* QuantizeLinear::_get_device() {
        
        return device;
    }
    
    void QuantizeLinear::init() {      
  
    }
    
    void QuantizeLinear::bind(std::string _x_input, std::string _y_scale_input, std::string _y_zero_point_input_opt, std::string _y_output){
        x_input = _x_input; y_scale_input = _y_scale_input; y_zero_point_input_opt = _y_zero_point_input_opt; y_output = _y_output;
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input_opt = tensor_dict[y_zero_point_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input_opt]->data(), *tensor_dict[y_output]->data());
    }



}



