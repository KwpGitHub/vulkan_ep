#include "size.h"
//cpp stuff
namespace layers {    
   
    Size::Size(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/size.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Size::_get_device() {        
        return backend::device;
    }
    
    void Size::init() {      
  
    }
    
    void Size::bind(std::string _data_i, std::string _size_o){
        data_i = _data_i; size_o = _size_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.size_o = backend::tensor_dict[size_o]->shape();
 
        
    }

    void Size::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[size_o]->data());
    }

    void Size::forward(){ 
        program->run();
    }

}

