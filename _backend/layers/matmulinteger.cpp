#include "MatMulInteger.h"
//cpp stuff
namespace backend {    
   
    MatMulInteger::MatMulInteger(std::string name) : Layer(name) { }
       
    vuh::Device* MatMulInteger::_get_device() {
        
        return device;
    }
    
    void MatMulInteger::init() {      
  
    }
    
    void MatMulInteger::bind(std::string _A_i, std::string _B_i, std::string _a_zero_point_i, std::string _b_zero_point_i, std::string _Y_o){
        A_i = _A_i; B_i = _B_i; a_zero_point_i = _a_zero_point_i; b_zero_point_i = _b_zero_point_i; Y_o = _Y_o;

		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
  		binding.a_zero_point_i = tensor_dict[a_zero_point_i]->shape();
  		binding.b_zero_point_i = tensor_dict[b_zero_point_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        
    }
}

