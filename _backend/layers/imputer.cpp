#include "imputer.h"
//cpp stuff
namespace layers {    
   
    Imputer::Imputer(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/imputer.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Imputer::_get_device() {        
        return backend::device;
    }
    
    void Imputer::init( std::vector<float> _imputed_value_floats,  std::vector<int> _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64) {      
		 imputed_value_floats = _imputed_value_floats; 
 		 imputed_value_int64s = _imputed_value_int64s; 
 		 replaced_value_float = _replaced_value_float; 
 		 replaced_value_int64 = _replaced_value_int64; 
  
    }
    
    void Imputer::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.imputed_value_floats = imputed_value_floats;
  		//binding.imputed_value_int64s = imputed_value_int64s;
  		//binding.replaced_value_float = replaced_value_float;
  		//binding.replaced_value_int64 = replaced_value_int64;
         
    }

    void Imputer::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void Imputer::forward(){ 
        //program->run();
    }

}

