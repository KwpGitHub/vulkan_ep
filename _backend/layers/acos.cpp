#include "acos.h"
//cpp stuff
namespace layers {    
   
    Acos::Acos(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/acos.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Acos::_get_device() {
        
        return backend::device;
    }
    
    void Acos::init() {      
  
    }
    
    void Acos::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
        
    }

    void Acos::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

