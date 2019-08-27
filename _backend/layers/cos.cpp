#include "cos.h"
//cpp stuff
namespace layers {    
   
    Cos::Cos(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/cos.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Cos::_get_device() {        
        return backend::device;
    }
    
    void Cos::init() {      
  
    }
    
    void Cos::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
        
    }

    void Cos::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void Cos::forward(){ 
        //program->run();
    }

}

