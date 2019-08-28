#include "mod.h"
//cpp stuff
namespace layers {    
   
    Mod::Mod(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/mod.spv");       
        dev = backend::device;
    }
       
        
    void Mod::init( int _fmod) {      
		 fmod = _fmod; 
  

    }
    
    void Mod::bind(std::string _A_i, std::string _B_i, std::string _C_o){    
        A_i = _A_i; B_i = _B_i; C_o = _C_o;        
		SHAPES.push_back(backend::tensor_dict[A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[C_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Mod::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[A_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[C_o]->data);
    }

    void Mod::forward(){ 
        program->run();
    }

}

