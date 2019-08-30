#include "if.h"
//cpp stuff
namespace layers {    
   
    If::If(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/if.spv");       
        dev = backend::g_device;
    }
       
        
    void If::init( int _else_branch,  int _then_branch) {      
		 m_else_branch = _else_branch; 
 		 m_then_branch = _then_branch; 
  

    }
    
    void If::bind(std::string _cond_i){    
        m_cond_i = _cond_i;        
		SHAPES.push_back(backend::tensor_dict[m_cond_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void If::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_cond_i]->data);
    }

    void If::forward(){ 
        program->run();
    }

}

