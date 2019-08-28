#include "scaler.h"
//cpp stuff
namespace layers {    
   
    Scaler::Scaler(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/scaler.spv");       
        dev = backend::device;
    }
       
        
    void Scaler::init( std::vector<float> _offset,  std::vector<float> _scale) {      
		 offset = _offset; 
 		 scale = _scale; 
  

    }
    
    void Scaler::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Scaler::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void Scaler::forward(){ 
        program->run();
    }

}

