#include "squeeze.h"
//cpp stuff
namespace layers {    
   
    Squeeze::Squeeze(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/squeeze.spv");       
        dev = backend::device;
    }
       
        
    void Squeeze::init( std::vector<int> _axes) {      
		 axes = _axes; 
  

    }
    
    void Squeeze::bind(std::string _data_i, std::string _squeezed_o){    
        data_i = _data_i; squeezed_o = _squeezed_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[squeezed_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Squeeze::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[data_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[data_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[data_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[squeezed_o]->data);
    }

    void Squeeze::forward(){ 
        program->run();
    }

}

