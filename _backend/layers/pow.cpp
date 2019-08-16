#include "Pow.h"

//cpp stuff
namespace backend {    
   
    Pow::Pow(std::string n) : Layer(n) { }
       
    vuh::Device* Pow::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Pow::init() {      
  
    }
    
    void Pow::bind(std::string _X_input, std::string _Y_input, std::string _Z_output){
        X_input = _X_input; Y_input = _Y_input; Z_output = _Z_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pow.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[Z_output]->data());
    }
    
}

    //backend::nn;

//python stuff


