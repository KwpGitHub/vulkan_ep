#include "qlinearmatmul.h"
//cpp stuff
namespace layers {    
   
    QLinearMatMul::QLinearMatMul(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/qlinearmatmul.spv");       
        dev = backend::device;
    }
       
        
    void QLinearMatMul::init() {      
  

    }
    
    void QLinearMatMul::bind(std::string _a_i, std::string _a_scale_i, std::string _a_zero_point_i, std::string _b_i, std::string _b_scale_i, std::string _b_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){    
        a_i = _a_i; a_scale_i = _a_scale_i; a_zero_point_i = _a_zero_point_i; b_i = _b_i; b_scale_i = _b_scale_i; b_zero_point_i = _b_zero_point_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[a_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[a_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[a_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[b_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[b_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[b_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QLinearMatMul::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[a_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[a_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[a_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[a_i]->data, *backend::tensor_dict[a_scale_i]->data, *backend::tensor_dict[a_zero_point_i]->data, *backend::tensor_dict[b_i]->data, *backend::tensor_dict[b_scale_i]->data, *backend::tensor_dict[b_zero_point_i]->data, *backend::tensor_dict[y_scale_i]->data, *backend::tensor_dict[y_zero_point_i]->data, *backend::tensor_dict[y_o]->data);
    }

    void QLinearMatMul::forward(){ 
        program->run();
    }

}

