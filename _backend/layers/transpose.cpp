#include "transpose.h"
//cpp stuff
namespace layers {    
   
    Transpose::Transpose(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/transpose.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Transpose::_get_device() {        
        return backend::device;
    }
    
    void Transpose::init( std::vector<int> _perm) {      
		 perm = _perm; 
  
    }
    
    void Transpose::bind(std::string _data_i, std::string _transposed_o){
        data_i = _data_i; transposed_o = _transposed_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.transposed_o = backend::tensor_dict[transposed_o]->shape();
 
		//binding.perm = perm;
         
    }

    void Transpose::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[transposed_o]->data());
    }

    void Transpose::forward(){ 
        //program->run();
    }

}

