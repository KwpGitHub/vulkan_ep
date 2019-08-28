#include "pad.h"
//cpp stuff
namespace layers {    
   
    Pad::Pad(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/pad.spv");       
        dev = backend::device;
    }
       
        
    void Pad::init( std::vector<int> _pads,  std::string _mode,  float _value) {      
		 pads = _pads; 
 		 mode = _mode; 
 		 value = _value; 
  

    }
    
    void Pad::bind(std::string _data_i, std::string _output_o){    
        data_i = _data_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Pad::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void Pad::forward(){ 
        program->run();
    }

}

