#include "OneHotEncoder.h"
//cpp stuff
namespace backend {    
   
    OneHotEncoder::OneHotEncoder() : Layer() { }
       
    vuh::Device* OneHotEncoder::_get_device() {
        
        return device;
    }
    
    void OneHotEncoder::init( Shape_t _cats_int64s,  int _zeros) {      
		 cats_int64s = _cats_int64s; 
 		 zeros = _zeros; 
  
    }
    
    void OneHotEncoder::bind(std::string _cats_strings, std::string _X_input, std::string _Y_output){
        cats_strings = _cats_strings; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.cats_int64s = cats_int64s;
  		binding.zeros = zeros;
 
		binding.cats_strings = tensor_dict[cats_strings]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehotencoder.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[cats_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



