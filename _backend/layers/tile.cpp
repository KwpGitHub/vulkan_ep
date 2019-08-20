#include "Tile.h"
//cpp stuff
namespace backend {    
   
    Tile::Tile(const std::string& name) : Layer(name) { }
       
    vuh::Device* Tile::_get_device() {
        
        return device;
    }
    
    void Tile::init() {      
  
    }
    
    void Tile::bind(std::string _input_i, std::string _repeats_i, std::string _output_o){
        input_i = _input_i; repeats_i = _repeats_i; output_o = _output_o;
		binding.input_i = tensor_dict[input_i]->shape();
  		binding.repeats_i = tensor_dict[repeats_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tile.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[repeats_i]->data(), *tensor_dict[output_o]->data());
    }

}

