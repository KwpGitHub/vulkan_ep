#include "instancenormalization.h"
//cpp stuff
namespace layers {    
   
    InstanceNormalization::InstanceNormalization(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\instancenormalization.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* InstanceNormalization::_get_device() {
        
        return backend::device;
    }
    
    void InstanceNormalization::init( float _epsilon) {      
		 epsilon = _epsilon; 
  
    }
    
    void InstanceNormalization::bind(std::string _input_i, std::string _scale_i, std::string _B_i, std::string _output_o){
        input_i = _input_i; scale_i = _scale_i; B_i = _B_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
  		//binding.scale_i = tensor_dict[scale_i]->shape();
  		//binding.B_i = tensor_dict[B_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.epsilon = epsilon;
         
    }

    void InstanceNormalization::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[scale_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[output_o]->data());
    }

}

