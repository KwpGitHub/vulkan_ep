#include "split.h"
//cpp stuff
namespace layers {    
   
    Split::Split(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/split.spv");       
        dev = backend::device;
    }
       
        
    void Split::init( int _axis,  std::vector<int> _split) {      
		 axis = _axis; 
 		 split = _split; 
  

    }
    
    void Split::bind(std::string _input_i){    
        input_i = _input_i;        
		SHAPES.push_back(backend::tensor_dict[input_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Split::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[input_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[input_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[input_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[input_i]->data);
    }

    void Split::forward(){ 
        program->run();
    }

}

