#include "categorymapper.h"
//cpp stuff
namespace layers {    
   
    CategoryMapper::CategoryMapper(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/categorymapper.spv");       
        dev = backend::device;
    }
       
        
    void CategoryMapper::init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _default_int64,  std::string _default_string) {      
		 cats_int64s = _cats_int64s; 
 		 cats_strings = _cats_strings; 
 		 default_int64 = _default_int64; 
 		 default_string = _default_string; 
  

    }
    
    void CategoryMapper::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void CategoryMapper::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void CategoryMapper::forward(){ 
        program->run();
    }

}

