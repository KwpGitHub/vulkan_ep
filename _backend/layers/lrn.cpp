#include "LRN.h"
//cpp stuff
namespace backend {    
   
    LRN::LRN() : Layer() { }
       
    vuh::Device* LRN::_get_device() {
        
        return device;
    }
    
    void LRN::init( int _size,  float _alpha,  float _beta,  float _bias) {      
		 size = _size; 
 		 alpha = _alpha; 
 		 beta = _beta; 
 		 bias = _bias; 
  
    }
    
    void LRN::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.size = size;
  		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.bias = bias;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lrn.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



