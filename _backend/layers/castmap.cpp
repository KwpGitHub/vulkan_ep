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
    
    void CastMap::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.cast_to = cast_to;
  		binding.map_form = map_form;
  		binding.max_map = max_map;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }

}

