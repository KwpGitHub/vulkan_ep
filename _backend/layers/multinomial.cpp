#include "multinomial.h"
//cpp stuff
namespace layers {    
   
    Multinomial::Multinomial(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/multinomial.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Multinomial::_get_device() {
        
        return backend::device;
    }
    
    void Multinomial::init( int _dtype,  int _sample_size,  float _seed) {      
		 dtype = _dtype; 
 		 sample_size = _sample_size; 
 		 seed = _seed; 
  
    }
    
    void Multinomial::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		//binding.input_i = tensor_dict[input_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.dtype = dtype;
  		//binding.sample_size = sample_size;
  		//binding.seed = seed;
         
    }

    void Multinomial::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

