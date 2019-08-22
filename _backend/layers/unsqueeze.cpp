#include "unsqueeze.h"
//cpp stuff
namespace layers {    
   
    Unsqueeze::Unsqueeze(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\unsqueeze.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Unsqueeze::_get_device() {
        
        return backend::device;
    }
    
    void Unsqueeze::init( std::vector<int> _axes) {      
		 axes = _axes; 
  
    }
    
    void Unsqueeze::bind(std::string _data_i, std::string _expanded_o){
        data_i = _data_i; expanded_o = _expanded_o;

		//binding.data_i = tensor_dict[data_i]->shape();
 
		//binding.expanded_o = tensor_dict[expanded_o]->shape();
 
		//binding.axes = axes;
         
    }

    void Unsqueeze::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[expanded_o]->data());
    }

}

