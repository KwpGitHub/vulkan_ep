#include "randomuniformlike.h"
//cpp stuff
namespace layers {    
   
    RandomUniformLike::RandomUniformLike(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/randomuniformlike.spv");       
        dev = backend::g_device;
    }
       
        
    void RandomUniformLike::init( int _dtype,  float _high,  float _low,  float _seed) {      
		 m_dtype = _dtype; 
 		 m_high = _high; 
 		 m_low = _low; 
 		 m_seed = _seed; 
  

    }
    
    void RandomUniformLike::bind(std::string _input_i, std::string _output_o){    
        m_input_i = _input_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RandomUniformLike::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_output_o]->data);
    }

    void RandomUniformLike::forward(){ 
        program->run();
    }

}

