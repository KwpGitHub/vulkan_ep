#include "onehotencoder.h"
//cpp stuff
namespace layers {    
   
    OneHotEncoder::OneHotEncoder(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/onehotencoder.spv");       
        dev = backend::device;
    }
       
        
    void OneHotEncoder::init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _zeros) {      
		 cats_int64s = _cats_int64s; 
 		 cats_strings = _cats_strings; 
 		 zeros = _zeros; 
  

    }
    
    void OneHotEncoder::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void OneHotEncoder::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void OneHotEncoder::forward(){ 
        program->run();
    }

}

