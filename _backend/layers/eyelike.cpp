#include "eyelike.h"
//cpp stuff
namespace layers {    
   
    EyeLike::EyeLike(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/eyelike.spv");       
        dev = backend::g_device;
    }
       
        
    void EyeLike::init( int _dtype,  int _k) {      
		 m_dtype = _dtype; 
 		 m_k = _k; 
  

    }
    
    void EyeLike::bind(std::string _input_i, std::string _output_o){    
        m_input_i = _input_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void EyeLike::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void EyeLike::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

