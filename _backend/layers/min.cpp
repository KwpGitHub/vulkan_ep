#include "min.h"
//cpp stuff
namespace layers {    
   
    Min::Min(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\min.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Min::_get_device() {
        
        return backend::device;
    }
    
    void Min::init() {      
  
    }
    
    void Min::bind(std::string _min_o){
        min_o = _min_o;


		//binding.min_o = tensor_dict[min_o]->shape();
 
        
    }

    void Min::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[min_o]->data());
    }

}

