#include "pow.h"
//cpp stuff
namespace layers {    
   
    Pow::Pow(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/pow.spv");       
        dev = backend::device;
    }
       
        
    void Pow::init() {      
  

    }
    
    void Pow::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){    
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[Y_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Pow::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_i]->data, *backend::tensor_dict[Z_o]->data);
    }

    void Pow::forward(){ 
        program->run();
    }

}

