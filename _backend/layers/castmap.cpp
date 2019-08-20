#include "CastMap.h"
//cpp stuff
namespace backend {    
   
    CastMap::CastMap(const std::string& name) : Layer(name) { }
       
    vuh::Device* CastMap::_get_device() {
        
        return device;
    }
    
    void CastMap::init( int _cast_to,  int _map_form,  int _max_map) {      
		 cast_to = _cast_to; 
 		 map_form = _map_form; 
 		 max_map = _max_map; 
  
    }
    
    void CastMap::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.cast_to = cast_to;
  		binding.map_form = map_form;
  		binding.max_map = max_map;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

