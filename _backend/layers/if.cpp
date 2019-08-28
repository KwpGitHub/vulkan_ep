#include "if.h"
//cpp stuff
namespace layers {    
   
    If::If(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/if.spv");       
        dev = backend::device;
    }
       
        
    void If::init( int _else_branch,  int _then_branch) {      
		 else_branch = _else_branch; 
 		 then_branch = _then_branch; 
  

    }
    
    void If::bind(std::string _cond_i){    
        cond_i = _cond_i;        
		SHAPES.push_back(backend::tensor_dict[cond_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void If::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[cond_i]->data);
    }

    void If::forward(){ 
        program->run();
    }

}

