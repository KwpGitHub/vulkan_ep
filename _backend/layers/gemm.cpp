#include "gemm.h"
//cpp stuff
namespace layers {    
   
    Gemm::Gemm(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/gemm.spv");       
        dev = backend::g_device;
    }
       
        
    void Gemm::init( float _alpha,  float _beta,  int _transA,  int _transB) {      
		 m_alpha = _alpha; 
 		 m_beta = _beta; 
 		 m_transA = _transA; 
 		 m_transB = _transB; 
  

    }
    
    void Gemm::bind(std::string _A_i, std::string _B_i, std::string _C_i, std::string _Y_o){    
        m_A_i = _A_i; m_B_i = _B_i; m_C_i = _C_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_C_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Gemm::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_A_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_C_i]->data, *backend::tensor_dict[m_Y_o]->data);       
    }

    void Gemm::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_A_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_C_i]->data, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

