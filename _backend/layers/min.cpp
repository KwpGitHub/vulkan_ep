#include "min.h"
//cpp stuff
namespace layers {    
   
    Min::Min(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/min.spv");       
        dev = backend::device;
    }
       
        
    void Min::init() {      
  

    }
    
    void Min::bind(std::string _min_o){    
        min_o = _min_o;        

		SHAPES.push_back(backend::tensor_dict[min_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Min::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[min_o]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[min_o]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[min_o]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[min_o]->data);
    }

    void Min::forward(){ 
        program->run();
    }

}

