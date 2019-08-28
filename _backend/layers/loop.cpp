#include "loop.h"
//cpp stuff
namespace layers {    
   
    Loop::Loop(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/loop.spv");       
        dev = backend::device;
    }
       
        
    void Loop::init( int _body) {      
		 body = _body; 
  

    }
    
    void Loop::bind(std::string _M_i, std::string _cond_i){    
        M_i = _M_i; cond_i = _cond_i;        
		SHAPES.push_back(backend::tensor_dict[M_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[cond_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Loop::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[M_i]->data, *backend::tensor_dict[cond_i]->data);
    }

    void Loop::forward(){ 
        program->run();
    }

}

