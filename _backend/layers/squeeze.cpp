#include "squeeze.h"
//cpp stuff
namespace layers {    
   
    Squeeze::Squeeze(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/squeeze.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Squeeze::_get_device() {        
        return backend::device;
    }
    
    void Squeeze::init( std::vector<int> _axes) {      
		 axes = _axes; 
  
    }
    
    void Squeeze::bind(std::string _data_i, std::string _squeezed_o){
        data_i = _data_i; squeezed_o = _squeezed_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.squeezed_o = backend::tensor_dict[squeezed_o]->shape();
 
		//binding.axes = axes;
         
    }

    void Squeeze::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[squeezed_o]->data());
    }

    void Squeeze::forward(){ 
        //program->run();
    }

}

