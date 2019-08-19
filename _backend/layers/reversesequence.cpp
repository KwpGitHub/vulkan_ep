#include "ReverseSequence.h"
//cpp stuff
namespace backend {    
   
    ReverseSequence::ReverseSequence() : Layer() { }
       
    vuh::Device* ReverseSequence::_get_device() {
        
        return device;
    }
    
    void ReverseSequence::init( int _batch_axis,  int _time_axis) {      
		 batch_axis = _batch_axis; 
 		 time_axis = _time_axis; 
  
    }
    
    void ReverseSequence::bind(std::string _input_input, std::string _sequence_lens_input, std::string _Y_output){
        input_input = _input_input; sequence_lens_input = _sequence_lens_input; Y_output = _Y_output;
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.sequence_lens_input = tensor_dict[sequence_lens_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.batch_axis = batch_axis;
  		binding.time_axis = time_axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reversesequence.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[sequence_lens_input]->data(), *tensor_dict[Y_output]->data());
    }



}



