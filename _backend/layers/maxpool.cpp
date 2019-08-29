#include "maxpool.h"
//cpp stuff
namespace layers {    
   
    MaxPool::MaxPool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/maxpool.spv");       
        dev = backend::device;
    }
       
        
    void MaxPool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  std::vector<int> _dilations,  std::vector<int> _pads,  int _storage_order,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 ceil_mode = _ceil_mode; 
 		 dilations = _dilations; 
 		 pads = _pads; 
 		 storage_order = _storage_order; 
 		 strides = _strides; 
  

    }
    
    void MaxPool::bind(std::string _X_i, std::string _Y_o, std::string _Indices_o){    
        X_i = _X_i; Y_o = _Y_o; Indices_o = _Indices_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MaxPool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Indices_o]->data);
    }

    void MaxPool::forward(){ 
        program->run();
    }

}

