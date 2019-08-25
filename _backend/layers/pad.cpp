#include "pad.h"
//cpp stuff
namespace layers {    
   
    Pad::Pad(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/pad.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Pad::_get_device() {
        
        return backend::device;
    }
    
    void Pad::init( std::vector<int> _pads,  std::string _mode,  float _value) {      
		 pads = _pads; 
 		 mode = _mode; 
 		 value = _value; 
  
    }
    
    void Pad::bind(std::string _data_i, std::string _output_o){
        data_i = _data_i; output_o = _output_o;

		//binding.data_i = tensor_dict[data_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.pads = pads;
  		//binding.mode = mode;
  		//binding.value = value;
         
    }

    void Pad::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[output_o]->data());
    }

}

