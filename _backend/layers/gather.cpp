#include "gather.h"
//cpp stuff
namespace layers {    
   
    Gather::Gather(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\gather.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Gather::_get_device() {
        
        return backend::device;
    }
    
    void Gather::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Gather::bind(std::string _data_i, std::string _indices_i, std::string _output_o){
        data_i = _data_i; indices_i = _indices_i; output_o = _output_o;

		//binding.data_i = tensor_dict[data_i]->shape();
  		//binding.indices_i = tensor_dict[indices_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.axis = axis;
         
    }

    void Gather::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[indices_i]->data(), *tensor_dict[output_o]->data());
    }

}

