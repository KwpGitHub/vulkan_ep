#include "equal.h"
//cpp stuff
namespace layers {    
   
    Equal::Equal(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/equal.spv");       
        dev = backend::device;
    }
       
        
    void Equal::init() {      
  

    }
    
    void Equal::bind(std::string _A_i, std::string _B_i, std::string _C_o){    
        A_i = _A_i; B_i = _B_i; C_o = _C_o;        
		SHAPES.push_back(backend::tensor_dict[A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[C_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Equal::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[A_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[A_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[A_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[A_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[C_o]->data);
    }

    void Equal::forward(){ 
        program->run();
    }

}

