#include "topk.h"
//cpp stuff
namespace layers {    
   
    TopK::TopK(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/topk.spv");       
        dev = backend::device;
    }
       
        
    void TopK::init( int _axis) {      
		 axis = _axis; 
  

    }
    
    void TopK::bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o){    
        X_i = _X_i; K_i = _K_i; Values_o = _Values_o; Indices_o = _Indices_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[K_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Values_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TopK::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[K_i]->data, *backend::tensor_dict[Values_o]->data, *backend::tensor_dict[Indices_o]->data);
    }

    void TopK::forward(){ 
        program->run();
    }

}

