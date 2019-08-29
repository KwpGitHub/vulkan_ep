#include "randomnormal.h"
//cpp stuff
namespace layers {    
   
    RandomNormal::RandomNormal(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/randomnormal.spv");       
        dev = backend::device;
    }
       
        
    void RandomNormal::init( std::vector<int> _shape,  int _dtype,  float _mean,  float _scale,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  

    }
    
    void RandomNormal::bind(std::string _output_o){    
        output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RandomNormal::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[output_o]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[output_o]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[output_o]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[output_o]->data);
    }

    void RandomNormal::forward(){ 
        program->run();
    }

}

