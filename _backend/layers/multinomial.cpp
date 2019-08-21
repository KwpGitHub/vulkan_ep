#include "Multinomial.h"
//cpp stuff
namespace backend {    
   
    Multinomial::Multinomial(std::string name) : Layer(name) { }
       
    vuh::Device* Multinomial::_get_device() {
        
        return device;
    }
    
    void Multinomial::init( int _dtype,  int _sample_size,  float _seed) {      
		 dtype = _dtype; 
 		 sample_size = _sample_size; 
 		 seed = _seed; 
  
    }
    
    void Multinomial::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.dtype = dtype;
  		binding.sample_size = sample_size;
  		binding.seed = seed;
 

        
    }
}

