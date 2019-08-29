#include "size.h"
//cpp stuff
namespace layers {    
   
    Size::Size(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/size.spv");       
        dev = backend::device;
    }
       
        
    void Size::init() {      
  

    }
    
    void Size::bind(std::string _data_i, std::string _size_o){    
        data_i = _data_i; size_o = _size_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[size_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Size::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[size_o]->data);
    }

    void Size::forward(){ 
        program->run();
    }

}

