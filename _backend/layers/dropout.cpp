#include "Dropout.h"

//cpp stuff
namespace backend {    
   
    Dropout::Dropout(std::string n) : Layer(n) { }
       
    vuh::Device* Dropout::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Dropout::init( float _ratio) {      
		 ratio = _ratio; 
  
    }
    
    void Dropout::bind(std::string _data_input, std::string _output_output, std::string _mask_output_opt){
        data_input = _data_input; output_output = _output_output; mask_output_opt = _mask_output_opt;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
  		binding.mask_output_opt = tensor_dict[mask_output_opt]->shape();
 
		binding.ratio = ratio;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dropout.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[output_output]->data(), *tensor_dict[mask_output_opt]->data());
    }
    
}

    //backend::nn;

//python stuff


