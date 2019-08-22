#include "expand.h"
//cpp stuff
namespace layers {    
   
    Expand::Expand(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\expand.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Expand::_get_device() {
        
        return backend::device;
    }
    
    void Expand::init() {      
  
    }
    
    void Expand::bind(std::string _input_i, std::string _shape_i, std::string _output_o){
        input_i = _input_i; shape_i = _shape_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
  		//binding.shape_i = tensor_dict[shape_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
        
    }

    void Expand::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[shape_i]->data(), *tensor_dict[output_o]->data());
    }

}

