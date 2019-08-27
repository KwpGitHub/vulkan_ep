#include "onehotencoder.h"
//cpp stuff
namespace layers {    
   
    OneHotEncoder::OneHotEncoder(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/onehotencoder.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* OneHotEncoder::_get_device() {        
        return backend::device;
    }
    
    void OneHotEncoder::init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _zeros) {      
		 cats_int64s = _cats_int64s; 
 		 cats_strings = _cats_strings; 
 		 zeros = _zeros; 
  
    }
    
    void OneHotEncoder::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.cats_int64s = cats_int64s;
  		//binding.cats_strings = cats_strings;
  		//binding.zeros = zeros;
         
    }

    void OneHotEncoder::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void OneHotEncoder::forward(){ 
        program->run();
    }

}

