#include "Size.h"
//cpp stuff
namespace backend {    
   
    Size::Size() : Layer() { }
       
    vuh::Device* Size::_get_device() {
        
        return device;
    }
    
    void Size::init() {      
  
    }
    
    void Size::bind(std::string _data_input, std::string _size_output){
        data_input = _data_input; size_output = _size_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.size_output = tensor_dict[size_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/size.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[size_output]->data());
    }



}



