#include "maxpool.h"
//cpp stuff
namespace layers {    
   
    MaxPool::MaxPool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/maxpool.spv");       
        dev = backend::g_device;
    }
       
        
    void MaxPool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  std::vector<int> _dilations,  std::vector<int> _pads,  int _storage_order,  std::vector<int> _strides) {      
		 m_kernel_shape = _kernel_shape; 
 		 m_auto_pad = _auto_pad; 
 		 m_ceil_mode = _ceil_mode; 
 		 m_dilations = _dilations; 
 		 m_pads = _pads; 
 		 m_storage_order = _storage_order; 
 		 m_strides = _strides; 
  

    }
    
    void MaxPool::bind(std::string _X_i, std::string _Y_o, std::string _Indices_o){    
        m_X_i = _X_i; m_Y_o = _Y_o; m_Indices_o = _Indices_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MaxPool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Indices_o]->data);
    }

    void MaxPool::forward(){ 
        program->run();
    }

}

