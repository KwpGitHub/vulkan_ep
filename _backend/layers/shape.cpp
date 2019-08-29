#include "shape.h"
//cpp stuff
namespace layers {    
   
    Shape::Shape(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/shape.spv");       
        dev = backend::device;
    }
       
        
    void Shape::init() {      
  

    }
    
    void Shape::bind(std::string _data_i, std::string _shape_o){    
        data_i = _data_i; shape_o = _shape_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[shape_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Shape::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[shape_o]->data);
    }

    void Shape::forward(){ 
        program->run();
    }

}

