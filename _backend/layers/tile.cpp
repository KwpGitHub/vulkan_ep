#include "tile.h"
//cpp stuff
namespace layers {    
   
    Tile::Tile(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/tile.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Tile::_get_device() {        
        return backend::device;
    }
    
    void Tile::init() {      
  
    }
    
    void Tile::bind(std::string _input_i, std::string _repeats_i, std::string _output_o){
        input_i = _input_i; repeats_i = _repeats_i; output_o = _output_o;

		binding.input_i = backend::tensor_dict[input_i]->shape();
  		binding.repeats_i = backend::tensor_dict[repeats_i]->shape();
 
		binding.output_o = backend::tensor_dict[output_o]->shape();
 
        
    }

    void Tile::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[input_i]->data(), *backend::tensor_dict[repeats_i]->data(), *backend::tensor_dict[output_o]->data());
    }

    void Tile::forward(){ 
        //program->run();
    }

}

