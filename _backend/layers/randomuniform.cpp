#include "randomuniform.h"
//cpp stuff
namespace layers {    
   
    RandomUniform::RandomUniform(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/randomuniform.spv");       
        dev = backend::device;
    }
       
        
    void RandomUniform::init( std::vector<int> _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  

    }
    
    void RandomUniform::bind(std::string _output_o){    
        output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RandomUniform::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[output_o]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[output_o]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[output_o]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[output_o]->data);
    }

    void RandomUniform::forward(){ 
        program->run();
    }

}

