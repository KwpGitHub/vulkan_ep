#include "ReverseSequence.h"
//cpp stuff
namespace backend {    
   
    ReverseSequence::ReverseSequence(std::string name) : Layer(name) { }
       
    vuh::Device* ReverseSequence::_get_device() {
        
        return device;
    }
    
    void ReverseSequence::init( int _batch_axis,  int _time_axis) {      
		 batch_axis = _batch_axis; 
 		 time_axis = _time_axis; 
  
    }
    
    void ReverseSequence::bind(std::string _input_i, std::string _sequence_lens_i, std::string _Y_o){
        input_i = _input_i; sequence_lens_i = _sequence_lens_i; Y_o = _Y_o;

		binding.input_i = tensor_dict[input_i]->shape();
  		binding.sequence_lens_i = tensor_dict[sequence_lens_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.batch_axis = batch_axis;
  		binding.time_axis = time_axis;
 

        
    }
}

