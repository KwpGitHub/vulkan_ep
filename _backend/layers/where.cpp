#include "Where.h"

//cpp stuff
namespace backend {    
   
    Where::Where(std::string n) : Layer(n) { }
       
    vuh::Device* Where::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Where::init() {      
  
    }
    
    void Where::bind(std::string _condition_input, std::string _X_input, std::string _Y_input, std::string _output_output){
        condition_input = _condition_input; X_input = _X_input; Y_input = _Y_input; output_output = _output_output;
		binding.condition_input = tensor_dict[condition_input]->shape();
  		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/where.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[condition_input]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


