#include "max.h"
//cpp stuff
namespace layers {    
   
    Max::Max(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/max.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Max::_get_device() {
        
        return backend::device;
    }
    
    void Max::init() {      
  
    }
    
    void Max::bind(std::string _max_o){
        max_o = _max_o;


		//binding.max_o = tensor_dict[max_o]->shape();
 
        
    }

    void Max::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[max_o]->data());
    }

}

