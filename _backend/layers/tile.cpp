#include "Tile.h"

//cpp stuff
namespace backend {    
   
    Tile::Tile(std::string n) : Layer(n) { }
       
    vuh::Device* Tile::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Tile::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.repeats_input = tensor_dict[repeats_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Tile::call(std::string input_input, std::string repeats_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tile.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[repeats_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


