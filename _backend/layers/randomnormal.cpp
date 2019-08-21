#include "RandomNormal.h"
//cpp stuff
namespace backend {    
   
    RandomNormal::RandomNormal(std::string name) : Layer(name) { }
       
    vuh::Device* RandomNormal::_get_device() {
        
        return device;
    }
    
    void RandomNormal::init( Shape_t _shape,  int _dtype,  float _mean,  float _scale,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  
    }
    
    void RandomNormal::bind(std::string _output_o){
        output_o = _output_o;


		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.mean = mean;
  		binding.scale = scale;
  		binding.seed = seed;
 

        
    }
}

