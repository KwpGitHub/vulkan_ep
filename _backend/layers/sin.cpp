#include "sin.h"
//cpp stuff
namespace layers {    
   
    Sin::Sin(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/sin.spv");       
        dev = backend::device;
    }
       
        
    void Sin::init() {      
  

    }
    
    void Sin::bind(std::string _input_i, std::string _output_o){    
        input_i = _input_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[input_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Sin::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[input_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[input_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[input_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[input_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Sin::forward(){ 
        program->run();
    }

}

