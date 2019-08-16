#include "Pad.h"

//cpp stuff
namespace backend {    
   
    Pad::Pad(std::string n) : Layer(n) { }
       
    vuh::Device* Pad::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Pad::init( Shape_t _pads,  int _mode,  float _value) {      
		 pads = _pads; 
 		 mode = _mode; 
 		 value = _value; 
  
    }
    
    void Pad::bind(std::string _data_input, std::string _output_output){
        data_input = _data_input; output_output = _output_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.pads = pads;
  		binding.mode = mode;
  		binding.value = value;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pad.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


