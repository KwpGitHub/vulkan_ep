#include "where.h"
//cpp stuff
namespace layers {    
   
    Where::Where(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/where.spv");       
        dev = backend::device;
    }
       
        
    void Where::init() {      
  

    }
    
    void Where::bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o){    
        condition_i = _condition_i; X_i = _X_i; Y_i = _Y_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[condition_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[Y_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Where::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[condition_i]->data, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Where::forward(){ 
        program->run();
    }

}

