#include "Scatter.h"

//cpp stuff
namespace backend {    
   
    Scatter::Scatter(std::string n) : Layer(n) { }
       
    vuh::Device* Scatter::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Scatter::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Scatter::bind(std::string _data_input, std::string _indices_input, std::string _updates_input, std::string _output_output){
        data_input = _data_input; indices_input = _indices_input; updates_input = _updates_input; output_output = _output_output;
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.indices_input = tensor_dict[indices_input]->shape();
  		binding.updates_input = tensor_dict[updates_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scatter.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[indices_input]->data(), *tensor_dict[updates_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


