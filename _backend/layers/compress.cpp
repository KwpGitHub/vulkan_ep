#include "compress.h"
//cpp stuff
namespace layers {    
   
    Compress::Compress(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/compress.spv");       
        dev = backend::g_device;
    }
       
        
    void Compress::init( int _axis) {      
		 m_axis = _axis; 
  

    }
    
    void Compress::bind(std::string _input_i, std::string _condition_i, std::string _output_o){    
        m_input_i = _input_i; m_condition_i = _condition_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_condition_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Compress::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_condition_i]->data, *backend::tensor_dict[m_output_o]->data);       
    }

    void Compress::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_input_i]->data, *backend::tensor_dict[m_condition_i]->data, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

