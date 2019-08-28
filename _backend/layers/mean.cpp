#include "mean.h"
//cpp stuff
namespace layers {    
   
    Mean::Mean(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/mean.spv");       
        dev = backend::device;
    }
       
        
    void Mean::init() {      
  

    }
    
    void Mean::bind(std::string _mean_o){    
        mean_o = _mean_o;        

		SHAPES.push_back(backend::tensor_dict[mean_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Mean::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[mean_o]->data);
    }

    void Mean::forward(){ 
        program->run();
    }

}

