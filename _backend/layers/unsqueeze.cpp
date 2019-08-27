#include "unsqueeze.h"
//cpp stuff
namespace layers {    
   
    Unsqueeze::Unsqueeze(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/unsqueeze.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Unsqueeze::_get_device() {        
        return backend::device;
    }
    
    void Unsqueeze::init( std::vector<int> _axes) {      
		 axes = _axes; 
  
    }
    
    void Unsqueeze::bind(std::string _data_i, std::string _expanded_o){
        data_i = _data_i; expanded_o = _expanded_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.expanded_o = backend::tensor_dict[expanded_o]->shape();
 
		//binding.axes = axes;
         
    }

    void Unsqueeze::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[expanded_o]->data());
    }

    void Unsqueeze::forward(){ 
        program->run();
    }

}

