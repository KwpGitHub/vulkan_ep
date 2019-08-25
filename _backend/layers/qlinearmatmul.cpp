#include "qlinearmatmul.h"
//cpp stuff
namespace layers {    
   
    QLinearMatMul::QLinearMatMul(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/qlinearmatmul.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* QLinearMatMul::_get_device() {
        
        return backend::device;
    }
    
    void QLinearMatMul::init() {      
  
    }
    
    void QLinearMatMul::bind(std::string _a_i, std::string _a_scale_i, std::string _a_zero_point_i, std::string _b_i, std::string _b_scale_i, std::string _b_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){
        a_i = _a_i; a_scale_i = _a_scale_i; a_zero_point_i = _a_zero_point_i; b_i = _b_i; b_scale_i = _b_scale_i; b_zero_point_i = _b_zero_point_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;

		//binding.a_i = tensor_dict[a_i]->shape();
  		//binding.a_scale_i = tensor_dict[a_scale_i]->shape();
  		//binding.a_zero_point_i = tensor_dict[a_zero_point_i]->shape();
  		//binding.b_i = tensor_dict[b_i]->shape();
  		//binding.b_scale_i = tensor_dict[b_scale_i]->shape();
  		//binding.b_zero_point_i = tensor_dict[b_zero_point_i]->shape();
  		//binding.y_scale_i = tensor_dict[y_scale_i]->shape();
  		//binding.y_zero_point_i = tensor_dict[y_zero_point_i]->shape();
 
		//binding.y_o = tensor_dict[y_o]->shape();
 
        
    }

    void QLinearMatMul::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[a_i]->data(), *tensor_dict[a_scale_i]->data(), *tensor_dict[a_zero_point_i]->data(), *tensor_dict[b_i]->data(), *tensor_dict[b_scale_i]->data(), *tensor_dict[b_zero_point_i]->data(), *tensor_dict[y_scale_i]->data(), *tensor_dict[y_zero_point_i]->data(), *tensor_dict[y_o]->data());
    }

}

