#include "SpaceToDepth.h"
//cpp stuff
namespace backend {    
   
    SpaceToDepth::SpaceToDepth(const std::string& name) : Layer(name) { }
       
    vuh::Device* SpaceToDepth::_get_device() {
        
        return device;
    }
    
    void SpaceToDepth::init( int _blocksize) {      
		 blocksize = _blocksize; 
  
    }
    
    void SpaceToDepth::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.blocksize = blocksize;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/spacetodepth.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }

}

