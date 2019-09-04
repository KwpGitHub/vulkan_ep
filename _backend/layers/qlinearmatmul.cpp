#include "qlinearmatmul.h"
//cpp stuff
namespace layers {    
   
    QLinearMatMul::QLinearMatMul(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/qlinearmatmul.spv");       
        dev = backend::g_device;
    }
       
        
    void QLinearMatMul::init() {      
  

    }
    
    void QLinearMatMul::bind(std::string _a_i, std::string _a_scale_i, std::string _a_zero_point_i, std::string _b_i, std::string _b_scale_i, std::string _b_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){    
        m_a_i = _a_i; m_a_scale_i = _a_scale_i; m_a_zero_point_i = _a_zero_point_i; m_b_i = _b_i; m_b_scale_i = _b_scale_i; m_b_zero_point_i = _b_zero_point_i; m_y_scale_i = _y_scale_i; m_y_zero_point_i = _y_zero_point_i; m_y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[m_a_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_a_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_a_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_b_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_b_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_b_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QLinearMatMul::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void QLinearMatMul::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_a_i]->data, *backend::tensor_dict[m_a_scale_i]->data, *backend::tensor_dict[m_a_zero_point_i]->data, *backend::tensor_dict[m_b_i]->data, *backend::tensor_dict[m_b_scale_i]->data, *backend::tensor_dict[m_b_zero_point_i]->data, *backend::tensor_dict[m_y_scale_i]->data, *backend::tensor_dict[m_y_zero_point_i]->data, *backend::tensor_dict[m_y_o]->data);
        //program->run();
    }

}

