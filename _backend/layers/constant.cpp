#include "constant.h"
//cpp stuff
namespace layers {    
   
    Constant::Constant(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/constant.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Constant::_get_device() {        
        return backend::device;
    }
    
    void Constant::init( std::vector<float> _value) {      
		 value = _value; 
  
    }
    
    void Constant::bind(std::string _output_o){
        output_o = _output_o;


		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.value = value;
         
    }

    void Constant::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[output_o]->data());
    }

    void Constant::forward(){ 
        program->run();
    }

}

