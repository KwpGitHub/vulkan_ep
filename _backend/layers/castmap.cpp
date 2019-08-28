#include "castmap.h"
//cpp stuff
namespace layers {    
   
    CastMap::CastMap(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/castmap.spv");       
        dev = backend::device;
    }
       
        
    void CastMap::init( std::string _cast_to,  std::string _map_form,  int _max_map) {      
		 cast_to = _cast_to; 
 		 map_form = _map_form; 
 		 max_map = _max_map; 
  

    }
    
    void CastMap::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void CastMap::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void CastMap::forward(){ 
        program->run();
    }

}

