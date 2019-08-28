#include "reducemax.h"
//cpp stuff
namespace layers {    
   
    ReduceMax::ReduceMax(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/reducemax.spv");       
        dev = backend::device;
    }
       
        
    void ReduceMax::init( std::vector<int> _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  

    }
    
    void ReduceMax::bind(std::string _data_i, std::string _reduced_o){    
        data_i = _data_i; reduced_o = _reduced_o;        
		SHAPES.push_back(backend::tensor_dict[data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[reduced_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ReduceMax::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[data_i]->data, *backend::tensor_dict[reduced_o]->data);
    }

    void ReduceMax::forward(){ 
        program->run();
    }

}

