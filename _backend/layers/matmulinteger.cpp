#include "MatMulInteger.h"

//cpp stuff
namespace backend {    
   
    MatMulInteger::MatMulInteger(std::string n) : Layer(n) { }
       
    vuh::Device* MatMulInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MatMulInteger::init() {      
    
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.a_zero_point_input_opt = tensor_dict[a_zero_point_input_opt]->shape();
  		binding.b_zero_point_input_opt = tensor_dict[b_zero_point_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 

    }
    
    void MatMulInteger::call(std::string A_input, std::string B_input, std::string a_zero_point_input_opt, std::string b_zero_point_input_opt, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[a_zero_point_input_opt]->data(), *tensor_dict[b_zero_point_input_opt]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


