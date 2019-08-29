#include "matmulinteger.h"
//cpp stuff
namespace layers {    
   
    MatMulInteger::MatMulInteger(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/matmulinteger.spv");       
        dev = backend::device;
    }
       
        
    void MatMulInteger::init() {      
  

    }
    
    void MatMulInteger::bind(std::string _A_i, std::string _B_i, std::string _a_zero_point_i, std::string _b_zero_point_i, std::string _Y_o){    
        A_i = _A_i; B_i = _B_i; a_zero_point_i = _a_zero_point_i; b_zero_point_i = _b_zero_point_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[a_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[b_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MatMulInteger::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[A_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[A_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[A_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[A_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[a_zero_point_i]->data, *backend::tensor_dict[b_zero_point_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void MatMulInteger::forward(){ 
        program->run();
    }

}

