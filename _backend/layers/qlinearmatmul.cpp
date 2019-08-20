#include "QLinearMatMul.h"
//cpp stuff
namespace backend {    
   
    QLinearMatMul::QLinearMatMul(const std::string& name) : Layer(name) { }
       
    vuh::Device* QLinearMatMul::_get_device() {
        
        return device;
    }
    
    void QLinearMatMul::init() {      
  
    }
    
    void QLinearMatMul::bind(std::string _a_i, std::string _a_scale_i, std::string _a_zero_point_i, std::string _b_i, std::string _b_scale_i, std::string _b_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){
        a_i = _a_i; a_scale_i = _a_scale_i; a_zero_point_i = _a_zero_point_i; b_i = _b_i; b_scale_i = _b_scale_i; b_zero_point_i = _b_zero_point_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;
		binding.a_i = tensor_dict[a_i]->shape();
  		binding.a_scale_i = tensor_dict[a_scale_i]->shape();
  		binding.a_zero_point_i = tensor_dict[a_zero_point_i]->shape();
  		binding.b_i = tensor_dict[b_i]->shape();
  		binding.b_scale_i = tensor_dict[b_scale_i]->shape();
  		binding.b_zero_point_i = tensor_dict[b_zero_point_i]->shape();
  		binding.y_scale_i = tensor_dict[y_scale_i]->shape();
  		binding.y_zero_point_i = tensor_dict[y_zero_point_i]->shape();
 
		binding.y_o = tensor_dict[y_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[a_i]->data(), *tensor_dict[a_scale_i]->data(), *tensor_dict[a_zero_point_i]->data(), *tensor_dict[b_i]->data(), *tensor_dict[b_scale_i]->data(), *tensor_dict[b_zero_point_i]->data(), *tensor_dict[y_scale_i]->data(), *tensor_dict[y_zero_point_i]->data(), *tensor_dict[y_o]->data());
    }

}
