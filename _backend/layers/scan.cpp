#include "scan.h"
//cpp stuff
namespace layers {    
   
    Scan::Scan(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/scan.spv");       
        dev = backend::g_device;
    }
       
        
    void Scan::init( int _body,  int _num_scan_inputs,  std::vector<int> _scan_input_axes,  std::vector<int> _scan_input_directions,  std::vector<int> _scan_output_axes,  std::vector<int> _scan_output_directions) {      
		 m_body = _body; 
 		 m_num_scan_inputs = _num_scan_inputs; 
 		 m_scan_input_axes = _scan_input_axes; 
 		 m_scan_input_directions = _scan_input_directions; 
 		 m_scan_output_axes = _scan_output_axes; 
 		 m_scan_output_directions = _scan_output_directions; 
  

    }
    
    void Scan::bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _y0_o, std::string _y1_o, std::string _y2_o, std::string _y3_o, std::string _y4_o, std::string _y5_o, std::string _y6_o, std::string _y7_o, std::string _y8_o, std::string _y9_o, std::string _y10_o, std::string _y11_o, std::string _y12_o, std::string _y13_o, std::string _y14_o, std::string _y15_o, std::string _y16_o, std::string _y17_o, std::string _y18_o, std::string _y19_o, std::string _y20_o, std::string _y21_o, std::string _y22_o, std::string _y23_o, std::string _y24_o, std::string _y25_o, std::string _y26_o, std::string _y27_o, std::string _y28_o, std::string _y29_o, std::string _y30_o, std::string _y31_o){    
        m_x0_i = _x0_i; m_x1_i = _x1_i; m_x2_i = _x2_i; m_x3_i = _x3_i; m_x4_i = _x4_i; m_x5_i = _x5_i; m_x6_i = _x6_i; m_x7_i = _x7_i; m_x8_i = _x8_i; m_x9_i = _x9_i; m_x10_i = _x10_i; m_x11_i = _x11_i; m_x12_i = _x12_i; m_x13_i = _x13_i; m_x14_i = _x14_i; m_x15_i = _x15_i; m_x16_i = _x16_i; m_x17_i = _x17_i; m_x18_i = _x18_i; m_x19_i = _x19_i; m_x20_i = _x20_i; m_x21_i = _x21_i; m_x22_i = _x22_i; m_x23_i = _x23_i; m_x24_i = _x24_i; m_x25_i = _x25_i; m_x26_i = _x26_i; m_x27_i = _x27_i; m_x28_i = _x28_i; m_x29_i = _x29_i; m_x30_i = _x30_i; m_x31_i = _x31_i; m_y0_o = _y0_o; m_y1_o = _y1_o; m_y2_o = _y2_o; m_y3_o = _y3_o; m_y4_o = _y4_o; m_y5_o = _y5_o; m_y6_o = _y6_o; m_y7_o = _y7_o; m_y8_o = _y8_o; m_y9_o = _y9_o; m_y10_o = _y10_o; m_y11_o = _y11_o; m_y12_o = _y12_o; m_y13_o = _y13_o; m_y14_o = _y14_o; m_y15_o = _y15_o; m_y16_o = _y16_o; m_y17_o = _y17_o; m_y18_o = _y18_o; m_y19_o = _y19_o; m_y20_o = _y20_o; m_y21_o = _y21_o; m_y22_o = _y22_o; m_y23_o = _y23_o; m_y24_o = _y24_o; m_y25_o = _y25_o; m_y26_o = _y26_o; m_y27_o = _y27_o; m_y28_o = _y28_o; m_y29_o = _y29_o; m_y30_o = _y30_o; m_y31_o = _y31_o;        
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
 
		SHAPES.push_back(backend::tensor_dict[m_y0_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y1_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y2_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y3_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y4_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y5_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y6_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y7_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y8_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y9_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y10_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y11_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y12_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y13_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y14_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y15_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y16_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y17_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y18_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y19_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y20_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y21_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y22_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y23_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y24_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y25_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y26_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y27_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y28_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y29_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y30_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y31_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Scan::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Scan::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_x0_i]->data, *backend::tensor_dict[m_x1_i]->data, *backend::tensor_dict[m_x2_i]->data, *backend::tensor_dict[m_x3_i]->data, *backend::tensor_dict[m_x4_i]->data, *backend::tensor_dict[m_x5_i]->data, *backend::tensor_dict[m_x6_i]->data, *backend::tensor_dict[m_x7_i]->data, *backend::tensor_dict[m_x8_i]->data, *backend::tensor_dict[m_x9_i]->data, *backend::tensor_dict[m_x10_i]->data, *backend::tensor_dict[m_x11_i]->data, *backend::tensor_dict[m_x12_i]->data, *backend::tensor_dict[m_x13_i]->data, *backend::tensor_dict[m_x14_i]->data, *backend::tensor_dict[m_x15_i]->data, *backend::tensor_dict[m_x16_i]->data, *backend::tensor_dict[m_x17_i]->data, *backend::tensor_dict[m_x18_i]->data, *backend::tensor_dict[m_x19_i]->data, *backend::tensor_dict[m_x20_i]->data, *backend::tensor_dict[m_x21_i]->data, *backend::tensor_dict[m_x22_i]->data, *backend::tensor_dict[m_x23_i]->data, *backend::tensor_dict[m_x24_i]->data, *backend::tensor_dict[m_x25_i]->data, *backend::tensor_dict[m_x26_i]->data, *backend::tensor_dict[m_x27_i]->data, *backend::tensor_dict[m_x28_i]->data, *backend::tensor_dict[m_x29_i]->data, *backend::tensor_dict[m_x30_i]->data, *backend::tensor_dict[m_x31_i]->data, *backend::tensor_dict[m_y0_o]->data, *backend::tensor_dict[m_y1_o]->data, *backend::tensor_dict[m_y2_o]->data, *backend::tensor_dict[m_y3_o]->data, *backend::tensor_dict[m_y4_o]->data, *backend::tensor_dict[m_y5_o]->data, *backend::tensor_dict[m_y6_o]->data, *backend::tensor_dict[m_y7_o]->data, *backend::tensor_dict[m_y8_o]->data, *backend::tensor_dict[m_y9_o]->data, *backend::tensor_dict[m_y10_o]->data, *backend::tensor_dict[m_y11_o]->data, *backend::tensor_dict[m_y12_o]->data, *backend::tensor_dict[m_y13_o]->data, *backend::tensor_dict[m_y14_o]->data, *backend::tensor_dict[m_y15_o]->data, *backend::tensor_dict[m_y16_o]->data, *backend::tensor_dict[m_y17_o]->data, *backend::tensor_dict[m_y18_o]->data, *backend::tensor_dict[m_y19_o]->data, *backend::tensor_dict[m_y20_o]->data, *backend::tensor_dict[m_y21_o]->data, *backend::tensor_dict[m_y22_o]->data, *backend::tensor_dict[m_y23_o]->data, *backend::tensor_dict[m_y24_o]->data, *backend::tensor_dict[m_y25_o]->data, *backend::tensor_dict[m_y26_o]->data, *backend::tensor_dict[m_y27_o]->data, *backend::tensor_dict[m_y28_o]->data, *backend::tensor_dict[m_y29_o]->data, *backend::tensor_dict[m_y30_o]->data, *backend::tensor_dict[m_y31_o]->data);
        //program->run();
    }

}

