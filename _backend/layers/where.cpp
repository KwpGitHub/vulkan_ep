#include "where.h"
//cpp stuff
namespace layers {    
   
    Where::Where(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/where.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Where::_get_device() {        
        return backend::device;
    }
    
    void Where::init() {      
  
    }
    
    void Where::bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o){
        condition_i = _condition_i; X_i = _X_i; Y_i = _Y_i; output_o = _output_o;

		binding.condition_i = backend::tensor_dict[condition_i]->shape();
  		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.Y_i = backend::tensor_dict[Y_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
        
    }

    void Where::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[condition_i]->data(), *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void Where::forward(){ 
        //program->run();
    }

}

