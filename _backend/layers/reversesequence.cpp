#include "reversesequence.h"
//cpp stuff
namespace layers {    
   
    ReverseSequence::ReverseSequence(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/reversesequence.spv");       
        dev = backend::g_device;
    }
       
        
    void ReverseSequence::init( int _batch_axis,  int _time_axis) {      
		 m_batch_axis = _batch_axis; 
 		 m_time_axis = _time_axis; 
  

    }
    
    void ReverseSequence::bind(std::string _input_i, std::string _sequence_lens_i, std::string _Y_o){    
        m_input_i = _input_i; m_sequence_lens_i = _sequence_lens_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_sequence_lens_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ReverseSequence::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_sequence_lens_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void ReverseSequence::forward(){ 
        program->run();
    }

}

