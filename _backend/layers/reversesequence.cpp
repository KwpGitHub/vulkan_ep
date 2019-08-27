#include "reversesequence.h"
//cpp stuff
namespace layers {    
   
    ReverseSequence::ReverseSequence(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/reversesequence.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ReverseSequence::_get_device() {        
        return backend::device;
    }
    
    void ReverseSequence::init( int _batch_axis,  int _time_axis) {      
		 batch_axis = _batch_axis; 
 		 time_axis = _time_axis; 
  
    }
    
    void ReverseSequence::bind(std::string _input_i, std::string _sequence_lens_i, std::string _Y_o){
        input_i = _input_i; sequence_lens_i = _sequence_lens_i; Y_o = _Y_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
  		binding.sequence_lens_i = backend::tensor_dict[sequence_lens_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.batch_axis = batch_axis;
  		//binding.time_axis = time_axis;
         
    }

    void ReverseSequence::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[sequence_lens_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void ReverseSequence::forward(){ 
        //program->run();
    }

}

