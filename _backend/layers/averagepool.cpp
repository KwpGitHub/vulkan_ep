#include "averagepool.h"
//cpp stuff
namespace layers {    
   
    AveragePool::AveragePool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/averagepool.spv");       
        dev = backend::device;
    }
       
        
    void AveragePool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  int _count_include_pad,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 ceil_mode = _ceil_mode; 
 		 count_include_pad = _count_include_pad; 
 		 pads = _pads; 
 		 strides = _strides; 
  

    }
    
    void AveragePool::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void AveragePool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void AveragePool::forward(){ 
        program->run();
    }

}

