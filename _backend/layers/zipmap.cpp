#include "ZipMap.h"
//cpp stuff
namespace backend {    
   
    ZipMap::ZipMap(const std::string& name) : Layer(name) { }
       
    vuh::Device* ZipMap::_get_device() {
        
        return device;
    }
    
    void ZipMap::init( Shape_t _classlabels_int64s) {      
		 classlabels_int64s = _classlabels_int64s; 
  
    }
    
    void ZipMap::bind(std::string _classlabels_strings, std::string _X_i, std::string _Z_o){
        classlabels_strings = _classlabels_strings; X_i = _X_i; Z_o = _Z_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Z_o = tensor_dict[Z_o]->shape();
 
		binding.classlabels_int64s = classlabels_int64s;
 
		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Z_o]->data());
    }

}

