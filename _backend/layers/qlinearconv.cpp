#include "qlinearconv.h"
//cpp stuff
namespace layers {    
   
    QLinearConv::QLinearConv(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/qlinearconv.spv");       
        dev = backend::g_device;
    }
       
        
    void QLinearConv::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 m_auto_pad = _auto_pad; 
 		 m_dilations = _dilations; 
 		 m_group = _group; 
 		 m_kernel_shape = _kernel_shape; 
 		 m_pads = _pads; 
 		 m_strides = _strides; 
  

    }
    
    void QLinearConv::bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _w_i, std::string _w_scale_i, std::string _w_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _B_i, std::string _y_o){    
        m_x_i = _x_i; m_x_scale_i = _x_scale_i; m_x_zero_point_i = _x_zero_point_i; m_w_i = _w_i; m_w_scale_i = _w_scale_i; m_w_zero_point_i = _w_zero_point_i; m_y_scale_i = _y_scale_i; m_y_zero_point_i = _y_zero_point_i; m_B_i = _B_i; m_y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[m_x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_w_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_w_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_w_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QLinearConv::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void QLinearConv::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_x_i]->data, *backend::tensor_dict[m_x_scale_i]->data, *backend::tensor_dict[m_x_zero_point_i]->data, *backend::tensor_dict[m_w_i]->data, *backend::tensor_dict[m_w_scale_i]->data, *backend::tensor_dict[m_w_zero_point_i]->data, *backend::tensor_dict[m_y_scale_i]->data, *backend::tensor_dict[m_y_zero_point_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_y_o]->data);
        //program->run();
    }

}

