#include "tile.h"
//cpp stuff
namespace layers {    
   
    Tile::Tile(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/tile.spv");       
        dev = backend::device;
    }
       
        
    void Tile::init() {      
  

    }
    
    void Tile::bind(std::string _input_i, std::string _repeats_i, std::string _output_o){    
        input_i = _input_i; repeats_i = _repeats_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[input_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[repeats_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Tile::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[input_i]->data, *backend::tensor_dict[repeats_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Tile::forward(){ 
        program->run();
    }

}

