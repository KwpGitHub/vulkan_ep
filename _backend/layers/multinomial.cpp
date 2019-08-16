#include "Multinomial.h"

//cpp stuff
namespace backend {    
   
    Multinomial::Multinomial(std::string n) : Layer(n) { }
       
    vuh::Device* Multinomial::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Multinomial::init( int _dtype,  int _sample_size,  float _seed) {      
		 dtype = _dtype; 
 		 sample_size = _sample_size; 
 		 seed = _seed; 
  
    }
    
    void Multinomial::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.sample_size = sample_size;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/multinomial.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


