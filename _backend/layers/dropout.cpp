#include "dropout.h"
//cpp stuff
namespace layers {    
   
    Dropout::Dropout(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/dropout.spv");       
        dev = backend::device;
    }
       
        
    void Dropout::init( float _ratio) {      
		 ratio = _ratio; 
  

    }
    
    void Dropout::bind(std::string _data_i, std::string _output_o, std::string _mask_o){    
        data_i = _data_i; output_o = _output_o; mask_o = _mask_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[mask_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Dropout::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[output_o]->data, *backend::tensor_dict[mask_o]->data);
    }

    void Dropout::forward(){ 
        program->run();
    }

}

