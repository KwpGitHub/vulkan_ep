#include "loop.h"
//cpp stuff
namespace layers {    
   
    Loop::Loop(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/loop.spv");       
        dev = backend::g_device;
    }
       
        
    void Loop::init( int _body) {      
		 m_body = _body; 
  

    }
    
    void Loop::bind(std::string _M_i, std::string _cond_i){    
        m_M_i = _M_i; m_cond_i = _cond_i;        
		SHAPES.push_back(backend::tensor_dict[m_M_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_cond_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Loop::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_M_i]->data, *backend::tensor_dict[m_cond_i]->data);
    }

    void Loop::forward(){ 
        program->run();
    }

}

