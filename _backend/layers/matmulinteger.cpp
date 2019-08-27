#include "matmulinteger.h"
//cpp stuff
namespace layers {    
   
    MatMulInteger::MatMulInteger(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/matmulinteger.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* MatMulInteger::_get_device() {        
        return backend::device;
    }
    
    void MatMulInteger::init() {      
  
    }
    
    void MatMulInteger::bind(std::string _A_i, std::string _B_i, std::string _a_zero_point_i, std::string _b_zero_point_i, std::string _Y_o){
        A_i = _A_i; B_i = _B_i; a_zero_point_i = _a_zero_point_i; b_zero_point_i = _b_zero_point_i; Y_o = _Y_o;

		binding.A_i = backend::tensor_dict[A_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
  		binding.a_zero_point_i = backend::tensor_dict[a_zero_point_i]->shape();
  		binding.b_zero_point_i = backend::tensor_dict[b_zero_point_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
        
    }

    void MatMulInteger::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[A_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[a_zero_point_i]->data(), *backend::tensor_dict[b_zero_point_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void MatMulInteger::forward(){ 
        program->run();
    }

}

