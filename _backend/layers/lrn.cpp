#include "lrn.h"
//cpp stuff
namespace layers {    
   
    LRN::LRN(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\lrn.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* LRN::_get_device() {
        
        return backend::device;
    }
    
    void LRN::init( int _size,  float _alpha,  float _beta,  float _bias) {      
		 size = _size; 
 		 alpha = _alpha; 
 		 beta = _beta; 
 		 bias = _bias; 
  
    }
    
    void LRN::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.size = size;
  		//binding.alpha = alpha;
  		//binding.beta = beta;
  		//binding.bias = bias;
         
    }

    void LRN::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

