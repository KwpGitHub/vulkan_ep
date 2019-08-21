#include "Gemm.h"
//cpp stuff
namespace backend {    
   
    Gemm::Gemm(std::string name) : Layer(name) { }
       
    vuh::Device* Gemm::_get_device() {
        
        return device;
    }
    
    void Gemm::init( float _alpha,  float _beta,  int _transA,  int _transB) {      
		 alpha = _alpha; 
 		 beta = _beta; 
 		 transA = _transA; 
 		 transB = _transB; 
  
    }
    
    void Gemm::bind(std::string _A_i, std::string _B_i, std::string _C_i, std::string _Y_o){
        A_i = _A_i; B_i = _B_i; C_i = _C_i; Y_o = _Y_o;

		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
  		binding.C_i = tensor_dict[C_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.transA = transA;
  		binding.transB = transB;
 

        
    }
}

