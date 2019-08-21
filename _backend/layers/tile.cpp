#include "Tile.h"
//cpp stuff
namespace backend {    
   
    Tile::Tile(std::string name) : Layer(name) { }
       
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
 


        
    }
}

