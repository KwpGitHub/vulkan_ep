#include "concat.h"
//cpp stuff
namespace layers {    
   
    Concat::Concat(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/concat.spv");       
        dev = backend::device;
    }
       
        
    void Concat::init( int _axis) {      
		 axis = _axis; 
  

    }
    
    void Concat::bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _concat_result_o){    
        x0_i = _x0_i; x1_i = _x1_i; x2_i = _x2_i; x3_i = _x3_i; x4_i = _x4_i; x5_i = _x5_i; x6_i = _x6_i; x7_i = _x7_i; x8_i = _x8_i; x9_i = _x9_i; x10_i = _x10_i; x11_i = _x11_i; x12_i = _x12_i; x13_i = _x13_i; x14_i = _x14_i; x15_i = _x15_i; x16_i = _x16_i; x17_i = _x17_i; x18_i = _x18_i; x19_i = _x19_i; x20_i = _x20_i; x21_i = _x21_i; x22_i = _x22_i; x23_i = _x23_i; x24_i = _x24_i; x25_i = _x25_i; x26_i = _x26_i; x27_i = _x27_i; x28_i = _x28_i; x29_i = _x29_i; x30_i = _x30_i; x31_i = _x31_i; concat_result_o = _concat_result_o;        
		SHAPES.push_back(backend::tensor_dict[x0_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x1_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x2_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x3_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x4_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x5_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x6_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x7_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x8_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x9_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x10_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x11_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x12_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x13_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x14_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x15_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x16_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x17_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x18_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x19_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x20_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x21_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x22_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x23_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x24_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x25_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x26_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x27_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x28_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x29_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x30_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x31_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[concat_result_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Concat::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[x0_i]->data, *backend::tensor_dict[x1_i]->data, *backend::tensor_dict[x2_i]->data, *backend::tensor_dict[x3_i]->data, *backend::tensor_dict[x4_i]->data, *backend::tensor_dict[x5_i]->data, *backend::tensor_dict[x6_i]->data, *backend::tensor_dict[x7_i]->data, *backend::tensor_dict[x8_i]->data, *backend::tensor_dict[x9_i]->data, *backend::tensor_dict[x10_i]->data, *backend::tensor_dict[x11_i]->data, *backend::tensor_dict[x12_i]->data, *backend::tensor_dict[x13_i]->data, *backend::tensor_dict[x14_i]->data, *backend::tensor_dict[x15_i]->data, *backend::tensor_dict[x16_i]->data, *backend::tensor_dict[x17_i]->data, *backend::tensor_dict[x18_i]->data, *backend::tensor_dict[x19_i]->data, *backend::tensor_dict[x20_i]->data, *backend::tensor_dict[x21_i]->data, *backend::tensor_dict[x22_i]->data, *backend::tensor_dict[x23_i]->data, *backend::tensor_dict[x24_i]->data, *backend::tensor_dict[x25_i]->data, *backend::tensor_dict[x26_i]->data, *backend::tensor_dict[x27_i]->data, *backend::tensor_dict[x28_i]->data, *backend::tensor_dict[x29_i]->data, *backend::tensor_dict[x30_i]->data, *backend::tensor_dict[x31_i]->data, *backend::tensor_dict[concat_result_o]->data);
    }

    void Concat::forward(){ 
        program->run();
    }

}

