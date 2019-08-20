#include "Size.h"
//cpp stuff
namespace backend {    
   
    Size::Size(const std::string& name) : Layer(name) { }
       
    vuh::Device* Size::_get_device() {
        
        return device;
    }
    
    void Size::init() {      
  
    }
    
    void Size::bind(std::string _data_i, std::string _size_o){
        data_i = _data_i; size_o = _size_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.size_o = tensor_dict[size_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/size.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[size_o]->data());
    }

}

