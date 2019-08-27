#include "randomnormal.h"
//cpp stuff
namespace layers {    
   
    RandomNormal::RandomNormal(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/randomnormal.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* RandomNormal::_get_device() {        
        return backend::device;
    }
    
    void RandomNormal::init( std::vector<int> _shape,  int _dtype,  float _mean,  float _scale,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  
    }
    
    void RandomNormal::bind(std::string _output_o){
        output_o = _output_o;


		binding.output_o = backend::tensor_dict[output_o]->shape();
 
		//binding.shape = shape;
  		//binding.dtype = dtype;
  		//binding.mean = mean;
  		//binding.scale = scale;
  		//binding.seed = seed;
         
    }

    void RandomNormal::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[output_o]->data());
    }

    void RandomNormal::forward(){ 
        program->run();
    }

}

