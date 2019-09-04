#include "multinomial.h"
//cpp stuff
namespace layers {    
   
    Multinomial::Multinomial(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/multinomial.spv");       
        dev = backend::g_device;
    }
       
        
    void Multinomial::init( int _dtype,  int _sample_size,  float _seed) {      
		 m_dtype = _dtype; 
 		 m_sample_size = _sample_size; 
 		 m_seed = _seed; 
  

    }
    
    void Multinomial::bind(std::string _input_i, std::string _output_o){    
        m_input_i = _input_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Multinomial::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Multinomial::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

