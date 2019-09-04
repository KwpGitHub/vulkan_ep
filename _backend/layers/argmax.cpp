#include "argmax.h"
//cpp stuff
namespace layers {    
   
    ArgMax::ArgMax(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/argmax.spv");       
        dev = backend::g_device;
    }
       
        
    void ArgMax::init( int _axis,  int _keepdims) {      
		 m_axis = _axis; 
 		 m_keepdims = _keepdims; 
  

    }
    
    void ArgMax::bind(std::string _data_i, std::string _reduced_o){    
        m_data_i = _data_i; m_reduced_o = _reduced_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_reduced_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ArgMax::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void ArgMax::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_reduced_o]->data);
        //program->run();
    }

}

