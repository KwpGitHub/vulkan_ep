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
  
    }
    
    void MatMulInteger::bind(std::string _A_input, std::string _B_input, std::string _a_zero_point_input_opt, std::string _b_zero_point_input_opt, std::string _Y_output){
        A_input = _A_input; B_input = _B_input; a_zero_point_input_opt = _a_zero_point_input_opt; b_zero_point_input_opt = _b_zero_point_input_opt; Y_output = _Y_output;
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.a_zero_point_input_opt = tensor_dict[a_zero_point_input_opt]->shape();
  		binding.b_zero_point_input_opt = tensor_dict[b_zero_point_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[a_zero_point_input_opt]->data(), *tensor_dict[b_zero_point_input_opt]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


