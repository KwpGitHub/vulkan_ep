#include "clip.h"
//cpp stuff
namespace layers {    
   
    Clip::Clip(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\clip.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Clip::_get_device() {
        
        return backend::device;
    }
    
    void Clip::init( float _max,  float _min) {      
		 max = _max; 
 		 min = _min; 
  
    }
    
    void Clip::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.max = max;
  		//binding.min = min;
         
    }

    void Clip::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

