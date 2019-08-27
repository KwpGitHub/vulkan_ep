#include "flatten.h"
//cpp stuff
namespace layers {    
   
    Flatten::Flatten(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/flatten.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Flatten::_get_device() {        
        return backend::device;
    }
    
    void Flatten::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Flatten::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.axis = axis;
         
    }

    void Flatten::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void Flatten::forward(){ 
        //program->run();
    }

}

