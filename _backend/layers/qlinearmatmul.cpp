#include "QLinearMatMul.h"
//cpp stuff
namespace backend {    
   
    QLinearMatMul::QLinearMatMul() : Layer() { }
       
    vuh::Device* QLinearMatMul::_get_device() {
        
        return device;
    }
    
    void QLinearMatMul::init() {      
  
    }
    
    void QLinearMatMul::bind(std::string _a_input, std::string _a_scale_input, std::string _a_zero_point_input, std::string _b_input, std::string _b_scale_input, std::string _b_zero_point_input, std::string _y_scale_input, std::string _y_zero_point_input, std::string _y_output){
        a_input = _a_input; a_scale_input = _a_scale_input; a_zero_point_input = _a_zero_point_input; b_input = _b_input; b_scale_input = _b_scale_input; b_zero_point_input = _b_zero_point_input; y_scale_input = _y_scale_input; y_zero_point_input = _y_zero_point_input; y_output = _y_output;
		binding.a_input = tensor_dict[a_input]->shape();
  		binding.a_scale_input = tensor_dict[a_scale_input]->shape();
  		binding.a_zero_point_input = tensor_dict[a_zero_point_input]->shape();
  		binding.b_input = tensor_dict[b_input]->shape();
  		binding.b_scale_input = tensor_dict[b_scale_input]->shape();
  		binding.b_zero_point_input = tensor_dict[b_zero_point_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input = tensor_dict[y_zero_point_input]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[a_input]->data(), *tensor_dict[a_scale_input]->data(), *tensor_dict[a_zero_point_input]->data(), *tensor_dict[b_input]->data(), *tensor_dict[b_scale_input]->data(), *tensor_dict[b_zero_point_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input]->data(), *tensor_dict[y_output]->data());
    }



}



