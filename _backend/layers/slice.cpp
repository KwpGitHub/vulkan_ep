#include "slice.h"
//cpp stuff
namespace layers {    
   
    Slice::Slice(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/slice.spv");       
        dev = backend::device;
    }
       
        
    void Slice::init() {      
  

    }
    
    void Slice::bind(std::string _data_i, std::string _starts_i, std::string _ends_i, std::string _axes_i, std::string _steps_i, std::string _output_o){    
        data_i = _data_i; starts_i = _starts_i; ends_i = _ends_i; axes_i = _axes_i; steps_i = _steps_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[starts_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[ends_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[axes_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[steps_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Slice::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[starts_i]->data, *backend::tensor_dict[ends_i]->data, *backend::tensor_dict[axes_i]->data, *backend::tensor_dict[steps_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Slice::forward(){ 
        program->run();
    }

}

