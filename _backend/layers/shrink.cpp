#include "shrink.h"
//cpp stuff
namespace layers {    
   
    Shrink::Shrink(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/shrink.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Shrink::_get_device() {        
        return backend::device;
    }
    
    void Shrink::init( float _bias,  float _lambd) {      
		 bias = _bias; 
 		 lambd = _lambd; 
  
    }
    
    void Shrink::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.bias = bias;
  		//binding.lambd = lambd;
         
    }

    void Shrink::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void Shrink::forward(){ 
        program->run();
    }

}

