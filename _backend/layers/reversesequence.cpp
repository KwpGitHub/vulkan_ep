#include "reversesequence.h"
//cpp stuff
namespace layers {    
   
    ReverseSequence::ReverseSequence(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/reversesequence.spv");       
        dev = backend::device;
    }
       
        
    void ReverseSequence::init( int _batch_axis,  int _time_axis) {      
		 batch_axis = _batch_axis; 
 		 time_axis = _time_axis; 
  

    }
    
    void ReverseSequence::bind(std::string _input_i, std::string _sequence_lens_i, std::string _Y_o){    
        input_i = _input_i; sequence_lens_i = _sequence_lens_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[input_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[sequence_lens_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ReverseSequence::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[input_i]->data, *backend::tensor_dict[sequence_lens_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void ReverseSequence::forward(){ 
        program->run();
    }

}

