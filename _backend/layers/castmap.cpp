#include "castmap.h"
//cpp stuff
namespace layers {    
   
    CastMap::CastMap(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/castmap.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* CastMap::_get_device() {        
        return backend::device;
    }
    
    void CastMap::init( std::string _cast_to,  std::string _map_form,  int _max_map) {      
		 cast_to = _cast_to; 
 		 map_form = _map_form; 
 		 max_map = _max_map; 
  
    }
    
    void CastMap::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.cast_to = cast_to;
  		//binding.map_form = map_form;
  		//binding.max_map = max_map;
         
    }

    void CastMap::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void CastMap::forward(){ 
        //program->run();
    }

}

