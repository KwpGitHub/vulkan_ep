#include "dropout.h"
//cpp stuff
namespace layers {    
   
    Dropout::Dropout(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/dropout.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Dropout::_get_device() {        
        return backend::device;
    }
    
    void Dropout::init( float _ratio) {      
		 ratio = _ratio; 
  
    }
    
    void Dropout::bind(std::string _data_i, std::string _output_o, std::string _mask_o){
        data_i = _data_i; output_o = _output_o; mask_o = _mask_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
  		binding.mask_o = backend::tensor_dict[mask_o]->shape();
 
		//binding.ratio = ratio;
         
    }

    void Dropout::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[output_o]->data(), *backend::tensor_dict[mask_o]->data());
    }

    void Dropout::forward(){ 
        program->run();
    }

}

