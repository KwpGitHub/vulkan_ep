#include "gemm.h"
//cpp stuff
namespace layers {    
   
    Gemm::Gemm(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/gemm.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Gemm::_get_device() {        
        return backend::device;
    }
    
    void Gemm::init( float _alpha,  float _beta,  int _transA,  int _transB) {      
		 alpha = _alpha; 
 		 beta = _beta; 
 		 transA = _transA; 
 		 transB = _transB; 
  
    }
    
    void Gemm::bind(std::string _A_i, std::string _B_i, std::string _C_i, std::string _Y_o){
        A_i = _A_i; B_i = _B_i; C_i = _C_i; Y_o = _Y_o;

		binding.A_i = backend::tensor_dict[A_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
  		binding.C_i = backend::tensor_dict[C_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.alpha = alpha;
  		//binding.beta = beta;
  		//binding.transA = transA;
  		//binding.transB = transB;
         
    }

    void Gemm::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[A_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[C_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void Gemm::forward(){ 
        //program->run();
    }

}

