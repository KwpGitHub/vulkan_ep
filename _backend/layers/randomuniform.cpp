#include "RandomUniform.h"
//cpp stuff
namespace backend {    
   
    RandomUniform::RandomUniform(std::string name) : Layer(name) { }
       
    vuh::Device* RandomUniform::_get_device() {
        
        return device;
    }
    
    void RandomUniform::init( Shape_t _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniform::bind(std::string _output_o){
        output_o = _output_o;


		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        
    }
}

