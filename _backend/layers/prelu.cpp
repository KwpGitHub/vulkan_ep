#include "prelu.h"
//cpp stuff
namespace layers {    
   
    PRelu::PRelu(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/prelu.spv");       
        dev = backend::device;
    }
       
        
    void PRelu::init() {      
  

    }
    
    void PRelu::bind(std::string _X_i, std::string _slope_i, std::string _Y_o){    
        X_i = _X_i; slope_i = _slope_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[slope_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void PRelu::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[slope_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void PRelu::forward(){ 
        program->run();
    }

}

