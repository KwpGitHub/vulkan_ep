#include "thresholdedrelu.h"
//cpp stuff
namespace layers {    
   
    ThresholdedRelu::ThresholdedRelu(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/thresholdedrelu.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ThresholdedRelu::_get_device() {
        
        return backend::device;
    }
    
    void ThresholdedRelu::init( float _alpha) {      
		 alpha = _alpha; 
  
    }
    
    void ThresholdedRelu::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.alpha = alpha;
         
    }

    void ThresholdedRelu::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

