#include "reshape.h"
//cpp stuff
namespace layers {    
   
    Reshape::Reshape(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/reshape.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Reshape::_get_device() {
        
        return backend::device;
    }
    
    void Reshape::init() {      
  
    }
    
    void Reshape::bind(std::string _data_i, std::string _shape_i, std::string _reshaped_o){
        data_i = _data_i; shape_i = _shape_i; reshaped_o = _reshaped_o;

		//binding.data_i = tensor_dict[data_i]->shape();
  		//binding.shape_i = tensor_dict[shape_i]->shape();
 
		//binding.reshaped_o = tensor_dict[reshaped_o]->shape();
 
        
    }

    void Reshape::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[shape_i]->data(), *tensor_dict[reshaped_o]->data());
    }

}

