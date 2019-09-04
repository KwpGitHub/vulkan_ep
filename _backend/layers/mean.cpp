#include "mean.h"
//cpp stuff
namespace layers {    
   
    Mean::Mean(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/mean.spv");       
        dev = backend::g_device;
    }
       
        
    void Mean::init() {      
  

    }
    
    void Mean::bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _mean_o){    
        m_x0_i = _x0_i; m_x1_i = _x1_i; m_x2_i = _x2_i; m_x3_i = _x3_i; m_x4_i = _x4_i; m_x5_i = _x5_i; m_x6_i = _x6_i; m_x7_i = _x7_i; m_x8_i = _x8_i; m_x9_i = _x9_i; m_x10_i = _x10_i; m_x11_i = _x11_i; m_x12_i = _x12_i; m_x13_i = _x13_i; m_x14_i = _x14_i; m_x15_i = _x15_i; m_x16_i = _x16_i; m_x17_i = _x17_i; m_x18_i = _x18_i; m_x19_i = _x19_i; m_x20_i = _x20_i; m_x21_i = _x21_i; m_x22_i = _x22_i; m_x23_i = _x23_i; m_x24_i = _x24_i; m_x25_i = _x25_i; m_x26_i = _x26_i; m_x27_i = _x27_i; m_x28_i = _x28_i; m_x29_i = _x29_i; m_x30_i = _x30_i; m_x31_i = _x31_i; m_mean_o = _mean_o;        
		SHAPES.push_back(backend::tensor_dict[m_x0_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x1_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x2_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x3_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x4_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x5_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x6_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x7_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x8_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x9_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x10_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x11_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x12_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x13_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x14_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x15_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x16_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x17_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x18_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x19_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x20_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x21_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x22_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x23_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x24_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x25_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x26_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x27_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x28_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x29_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x30_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_x31_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_mean_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Mean::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Mean::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_x0_i]->data, *backend::tensor_dict[m_x1_i]->data, *backend::tensor_dict[m_x2_i]->data, *backend::tensor_dict[m_x3_i]->data, *backend::tensor_dict[m_x4_i]->data, *backend::tensor_dict[m_x5_i]->data, *backend::tensor_dict[m_x6_i]->data, *backend::tensor_dict[m_x7_i]->data, *backend::tensor_dict[m_x8_i]->data, *backend::tensor_dict[m_x9_i]->data, *backend::tensor_dict[m_x10_i]->data, *backend::tensor_dict[m_x11_i]->data, *backend::tensor_dict[m_x12_i]->data, *backend::tensor_dict[m_x13_i]->data, *backend::tensor_dict[m_x14_i]->data, *backend::tensor_dict[m_x15_i]->data, *backend::tensor_dict[m_x16_i]->data, *backend::tensor_dict[m_x17_i]->data, *backend::tensor_dict[m_x18_i]->data, *backend::tensor_dict[m_x19_i]->data, *backend::tensor_dict[m_x20_i]->data, *backend::tensor_dict[m_x21_i]->data, *backend::tensor_dict[m_x22_i]->data, *backend::tensor_dict[m_x23_i]->data, *backend::tensor_dict[m_x24_i]->data, *backend::tensor_dict[m_x25_i]->data, *backend::tensor_dict[m_x26_i]->data, *backend::tensor_dict[m_x27_i]->data, *backend::tensor_dict[m_x28_i]->data, *backend::tensor_dict[m_x29_i]->data, *backend::tensor_dict[m_x30_i]->data, *backend::tensor_dict[m_x31_i]->data, *backend::tensor_dict[m_mean_o]->data);
        //program->run();
    }

}

