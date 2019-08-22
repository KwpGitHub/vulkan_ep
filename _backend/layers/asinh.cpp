#include "asinh.h"
//cpp stuff
namespace layers {    
   
    Asinh::Asinh(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\asinh.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Asinh::_get_device() {
        
        return backend::device;
    }
    
    void Asinh::init() {      
  
    }
    
    void Asinh::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
        
    }

    void Asinh::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

