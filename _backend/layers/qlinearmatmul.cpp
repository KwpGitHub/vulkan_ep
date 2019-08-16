#include "QLinearMatMul.h"

//cpp stuff
namespace backend {    
   
    QLinearMatMul::QLinearMatMul(std::string n) : Layer(n) { }
       
    vuh::Device* QLinearMatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QLinearMatMul::init() {      
    
		binding.a_input = tensor_dict[a_input]->shape();
  		binding.a_scale_input = tensor_dict[a_scale_input]->shape();
  		binding.a_zero_point_input = tensor_dict[a_zero_point_input]->shape();
  		binding.b_input = tensor_dict[b_input]->shape();
  		binding.b_scale_input = tensor_dict[b_scale_input]->shape();
  		binding.b_zero_point_input = tensor_dict[b_zero_point_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input = tensor_dict[y_zero_point_input]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 

    }
    
    void QLinearMatMul::call(std::string a_input, std::string a_scale_input, std::string a_zero_point_input, std::string b_input, std::string b_scale_input, std::string b_zero_point_input, std::string y_scale_input, std::string y_zero_point_input, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[a_input]->data(), *tensor_dict[a_scale_input]->data(), *tensor_dict[a_zero_point_input]->data(), *tensor_dict[b_input]->data(), *tensor_dict[b_scale_input]->data(), *tensor_dict[b_zero_point_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input]->data(), *tensor_dict[y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


