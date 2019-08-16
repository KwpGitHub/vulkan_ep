#include "Mean.h"

//cpp stuff
namespace backend {    
   
    Mean::Mean(std::string n) : Layer(n) { }
       
    vuh::Device* Mean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Mean::init() {      
  
    }
    
    void Mean::bind(std::string _mean_output){
        mean_output = _mean_output;

		binding.mean_output = tensor_dict[mean_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[mean_output]->data());
    }
    
}

    //backend::nn;

//python stuff


