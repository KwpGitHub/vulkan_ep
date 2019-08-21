#include "RandomUniformLike.h"
//cpp stuff
namespace backend {    
   
    RandomUniformLike::RandomUniformLike(std::string name) : Layer(name) { }
       
    vuh::Device* RandomUniformLike::_get_device() {
        
        return device;
    }
    
    void RandomUniformLike::init( int _dtype,  float _high,  float _low,  float _seed) {      
		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniformLike::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        
    }
}

