#include "unsqueeze.h"
//cpp stuff
namespace layers {    
   
    Unsqueeze::Unsqueeze(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/unsqueeze.spv");       
        dev = backend::device;
    }
       
        
    void Unsqueeze::init( std::vector<int> _axes) {      
		 axes = _axes; 
  

    }
    
    void Unsqueeze::bind(std::string _data_i, std::string _expanded_o){    
        data_i = _data_i; expanded_o = _expanded_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[expanded_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Unsqueeze::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[expanded_o]->data);
    }

    void Unsqueeze::forward(){ 
        program->run();
    }

}

