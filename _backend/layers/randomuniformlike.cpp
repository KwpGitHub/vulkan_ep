#include "randomuniformlike.h"
//cpp stuff
namespace layers {    
   
    RandomUniformLike::RandomUniformLike(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/randomuniformlike.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* RandomUniformLike::_get_device() {        
        return backend::device;
    }
    
    void RandomUniformLike::init( int _dtype,  float _high,  float _low,  float _seed) {      
		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniformLike::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.dtype = dtype;
  		//binding.high = high;
  		//binding.low = low;
  		//binding.seed = seed;
         
    }

    void RandomUniformLike::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void RandomUniformLike::forward(){ 
        //program->run();
    }

}

