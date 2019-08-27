#include "split.h"
//cpp stuff
namespace layers {    
   
    Split::Split(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/split.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Split::_get_device() {        
        return backend::device;
    }
    
    void Split::init( int _axis,  std::vector<int> _split) {      
		 axis = _axis; 
 		 split = _split; 
  
    }
    
    void Split::bind(std::string _input_i){
        input_i = _input_i;

		binding.input_i = backend::tensor_dict[input_i]->shape();
 

		//binding.axis = axis;
  		//binding.split = split;
         
    }

    void Split::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data());
    }

    void Split::forward(){ 
        program->run();
    }

}

