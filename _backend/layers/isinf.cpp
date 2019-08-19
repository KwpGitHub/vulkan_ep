#include "IsInf.h"
//cpp stuff
namespace backend {    
   
    IsInf::IsInf() : Layer() { }
       
    vuh::Device* IsInf::_get_device() {
        
        return device;
    }
    
    void IsInf::init( int _detect_negative,  int _detect_positive) {      
		 detect_negative = _detect_negative; 
 		 detect_positive = _detect_positive; 
  
    }
    
    void IsInf::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.detect_negative = detect_negative;
  		binding.detect_positive = detect_positive;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/isinf.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



