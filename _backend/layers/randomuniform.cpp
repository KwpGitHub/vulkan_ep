#include "randomuniform.h"
//cpp stuff
namespace layers {    
   
    RandomUniform::RandomUniform(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/randomuniform.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* RandomUniform::_get_device() {        
        return backend::device;
    }
    
    void RandomUniform::init( std::vector<int> _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniform::bind(std::string _output_o){
        output_o = _output_o;


		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.shape = shape;
  		//binding.dtype = dtype;
  		//binding.high = high;
  		//binding.low = low;
  		//binding.seed = seed;
         
    }

    void RandomUniform::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[output_o]->data());
    }

    void RandomUniform::forward(){ 
        //program->run();
    }

}

