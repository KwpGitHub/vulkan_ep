#include "onehot.h"
//cpp stuff
namespace layers {    
   
    OneHot::OneHot(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/onehot.spv");       
        dev = backend::device;
    }
       
        
    void OneHot::init( int _axis) {      
		 axis = _axis; 
  

    }
    
    void OneHot::bind(std::string _indices_i, std::string _depth_i, std::string _values_i, std::string _output_o){    
        indices_i = _indices_i; depth_i = _depth_i; values_i = _values_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[indices_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[depth_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[values_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void OneHot::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[indices_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[indices_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[indices_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[indices_i]->data, *backend::tensor_dict[depth_i]->data, *backend::tensor_dict[values_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void OneHot::forward(){ 
        program->run();
    }

}

