#include "QuantizeLinear.h"
//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(const std::string& name) : Layer(name) { }
       
    vuh::Device* QuantizeLinear::_get_device() {
        
        return device;
    }
    
    void QuantizeLinear::init() {      
  
    }
    
    void QuantizeLinear::bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){
        x_i = _x_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;
		binding.x_i = tensor_dict[x_i]->shape();
  		binding.y_scale_i = tensor_dict[y_scale_i]->shape();
  		binding.y_zero_point_i = tensor_dict[y_zero_point_i]->shape();
 
		binding.y_o = tensor_dict[y_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x_i]->data(), *tensor_dict[y_scale_i]->data(), *tensor_dict[y_zero_point_i]->data(), *tensor_dict[y_o]->data());
    }

}
