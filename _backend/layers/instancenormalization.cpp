#include "InstanceNormalization.h"
//cpp stuff
namespace backend {    
   
    InstanceNormalization::InstanceNormalization(std::string name) : Layer(name) { }
       
    vuh::Device* InstanceNormalization::_get_device() {
        
        return device;
    }
    
    void InstanceNormalization::init( float _epsilon) {      
		 epsilon = _epsilon; 
  
    }
    
    void InstanceNormalization::bind(std::string _input_i, std::string _scale_i, std::string _B_i, std::string _output_o){
        input_i = _input_i; scale_i = _scale_i; B_i = _B_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
  		binding.scale_i = tensor_dict[scale_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.epsilon = epsilon;
 

        
    }
}

