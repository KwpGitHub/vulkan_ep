#include "Pad.h"
//cpp stuff
namespace backend {    
   
    Pad::Pad(const std::string& name) : Layer(name) { }
       
    vuh::Device* Pad::_get_device() {
        
        return device;
    }
    
    void Pad::init( Shape_t _pads,  int _mode,  float _value) {      
		 pads = _pads; 
 		 mode = _mode; 
 		 value = _value; 
  
    }
    
    void Pad::bind(std::string _data_i, std::string _output_o){
        data_i = _data_i; output_o = _output_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.pads = pads;
  		binding.mode = mode;
  		binding.value = value;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pad.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[output_o]->data());
    }

}

