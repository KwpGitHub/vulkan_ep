#include "max.h"
//cpp stuff
namespace layers {    
   
    Max::Max(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/max.spv");       
        dev = backend::device;
    }
       
        
    void Max::init() {      
  

    }
    
    void Max::bind(std::string _max_o){    
        max_o = _max_o;        

		SHAPES.push_back(backend::tensor_dict[max_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Max::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[max_o]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[max_o]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[max_o]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[max_o]->data);
    }

    void Max::forward(){ 
        program->run();
    }

}

