#include "Gemm.h"
//cpp stuff
namespace backend {    
   
    Gemm::Gemm(const std::string& name) : Layer(name) { }
       
    vuh::Device* Gemm::_get_device() {
        
        return device;
    }
    
    void Gemm::init( float _alpha,  float _beta,  int _transA,  int _transB) {      
		 alpha = _alpha; 
 		 beta = _beta; 
 		 transA = _transA; 
 		 transB = _transB; 
  
    }
    
    void Gemm::bind(std::string _A_input, std::string _B_input, std::string _C_input, std::string _Y_output){
        A_input = _A_input; B_input = _B_input; C_input = _C_input; Y_output = _Y_output;
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.C_input = tensor_dict[C_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.transA = transA;
  		binding.transB = transB;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gemm.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[C_input]->data(), *tensor_dict[Y_output]->data());
    }

}

