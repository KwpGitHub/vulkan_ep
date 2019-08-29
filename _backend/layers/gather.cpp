#include "gather.h"
//cpp stuff
namespace layers {    
   
    Gather::Gather(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/gather.spv");       
        dev = backend::device;
    }
       
        
    void Gather::init( int _axis) {      
		 axis = _axis; 
  

    }
    
    void Gather::bind(std::string _data_i, std::string _indices_i, std::string _output_o){    
        data_i = _data_i; indices_i = _indices_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[indices_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Gather::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[indices_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Gather::forward(){ 
        program->run();
    }

}

