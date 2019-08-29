#include "maxunpool.h"
//cpp stuff
namespace layers {    
   
    MaxUnpool::MaxUnpool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/maxunpool.spv");       
        dev = backend::device;
    }
       
        
    void MaxUnpool::init( std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  

    }
    
    void MaxUnpool::bind(std::string _X_i, std::string _I_i, std::string _output_shape_i, std::string _output_o){    
        X_i = _X_i; I_i = _I_i; output_shape_i = _output_shape_i; output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[I_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[output_shape_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MaxUnpool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[I_i]->data, *backend::tensor_dict[output_shape_i]->data, *backend::tensor_dict[output_o]->data);
    }

    void MaxUnpool::forward(){ 
        program->run();
    }

}

