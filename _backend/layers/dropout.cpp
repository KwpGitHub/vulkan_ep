#include "Dropout.h"
//cpp stuff
namespace backend {    
   
    Dropout::Dropout(const std::string& name) : Layer(name) { }
       
    vuh::Device* Dropout::_get_device() {
        
        return device;
    }
    
    void Dropout::init( float _ratio) {      
		 ratio = _ratio; 
  
    }
    
    void Dropout::bind(std::string _data_i, std::string _output_o, std::string _mask_o){
        data_i = _data_i; output_o = _output_o; mask_o = _mask_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
  		binding.mask_o = tensor_dict[mask_o]->shape();
 
		binding.ratio = ratio;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dropout.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[output_o]->data(), *tensor_dict[mask_o]->data());
    }

}

