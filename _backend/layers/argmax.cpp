#include "argmax.h"
//cpp stuff
namespace layers {    
   
    ArgMax::ArgMax(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/argmax.spv");       
        dev = backend::device;
    }
       
        
    void ArgMax::init( int _axis,  int _keepdims) {      
		 axis = _axis; 
 		 keepdims = _keepdims; 
  

    }
    
    void ArgMax::bind(std::string _data_i, std::string _reduced_o){    
        data_i = _data_i; reduced_o = _reduced_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[reduced_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ArgMax::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[reduced_o]->data);
    }

    void ArgMax::forward(){ 
        program->run();
    }

}

