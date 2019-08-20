#include "Imputer.h"
//cpp stuff
namespace backend {    
   
    Imputer::Imputer(const std::string& name) : Layer(name) { }
       
    vuh::Device* Imputer::_get_device() {
        
        return device;
    }
    
    void Imputer::init( Shape_t _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64) {      
		 imputed_value_int64s = _imputed_value_int64s; 
 		 replaced_value_float = _replaced_value_float; 
 		 replaced_value_int64 = _replaced_value_int64; 
  
    }
    
    void Imputer::bind(std::string _imputed_value_floats, std::string _X_i, std::string _Y_o){
        imputed_value_floats = _imputed_value_floats; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.imputed_value_int64s = imputed_value_int64s;
  		binding.replaced_value_float = replaced_value_float;
  		binding.replaced_value_int64 = replaced_value_int64;
 
		binding.imputed_value_floats = tensor_dict[imputed_value_floats]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/imputer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[imputed_value_floats]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}
