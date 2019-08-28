#include "qlinearconv.h"
//cpp stuff
namespace layers {    
   
    QLinearConv::QLinearConv(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/qlinearconv.spv");       
        dev = backend::device;
    }
       
        
    void QLinearConv::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  

    }
    
    void QLinearConv::bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _w_i, std::string _w_scale_i, std::string _w_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _B_i, std::string _y_o){    
        x_i = _x_i; x_scale_i = _x_scale_i; x_zero_point_i = _x_zero_point_i; w_i = _w_i; w_scale_i = _w_scale_i; w_zero_point_i = _w_zero_point_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; B_i = _B_i; y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[w_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[w_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[w_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QLinearConv::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[x_i]->data, *backend::tensor_dict[x_scale_i]->data, *backend::tensor_dict[x_zero_point_i]->data, *backend::tensor_dict[w_i]->data, *backend::tensor_dict[w_scale_i]->data, *backend::tensor_dict[w_zero_point_i]->data, *backend::tensor_dict[y_scale_i]->data, *backend::tensor_dict[y_zero_point_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[y_o]->data);
    }

    void QLinearConv::forward(){ 
        program->run();
    }

}

