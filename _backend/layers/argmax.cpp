#include "argmax.h"
//cpp stuff
namespace layers {    
   
    ArgMax::ArgMax(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/argmax.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ArgMax::_get_device() {        
        return backend::device;
    }
    
    void ArgMax::init( int _axis,  int _keepdims) {      
		 axis = _axis; 
 		 keepdims = _keepdims; 
  
    }
    
    void ArgMax::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.reduced_o = backend::tensor_dict[reduced_o]->shape();
 
		//binding.axis = axis;
  		//binding.keepdims = keepdims;
         
    }

    void ArgMax::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[reduced_o]->data());
    }

    void ArgMax::forward(){ 
        program->run();
    }

}

