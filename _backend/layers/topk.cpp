#include "topk.h"
//cpp stuff
namespace layers {    
   
    TopK::TopK(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/topk.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* TopK::_get_device() {        
        return backend::device;
    }
    
    void TopK::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void TopK::bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o){
        X_i = _X_i; K_i = _K_i; Values_o = _Values_o; Indices_o = _Indices_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.K_i = backend::tensor_dict[K_i]->shape();
 
		binding.Values_o = backend::tensor_dict[Values_o]->shape();
  		binding.Indices_o = backend::tensor_dict[Indices_o]->shape();
 
		//binding.axis = axis;
         
    }

    void TopK::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[K_i]->data(), *backend::tensor_dict[Values_o]->data(), *backend::tensor_dict[Indices_o]->data());
    }

    void TopK::forward(){ 
        //program->run();
    }

}

