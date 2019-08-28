#include "constant.h"
//cpp stuff
namespace layers {    
   
    Constant::Constant(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/constant.spv");       
        dev = backend::device;
    }
       
        
    void Constant::init( std::vector<float> _value) {      
		 value = _value; 
  

    }
    
    void Constant::bind(std::string _output_o){    
        output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Constant::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[output_o]->data);
    }

    void Constant::forward(){ 
        program->run();
    }

}

