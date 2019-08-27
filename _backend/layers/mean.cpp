#include "mean.h"
//cpp stuff
namespace layers {    
   
    Mean::Mean(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/mean.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Mean::_get_device() {        
        return backend::device;
    }
    
    void Mean::init() {      
  
    }
    
    void Mean::bind(std::string _mean_o){
        mean_o = _mean_o;


		binding.mean_o = backend::tensor_dict[mean_o]->shape();
 
        
    }

    void Mean::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[mean_o]->data());
    }

    void Mean::forward(){ 
        //program->run();
    }

}

