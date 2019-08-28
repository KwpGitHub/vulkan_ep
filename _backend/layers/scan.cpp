#include "scan.h"
//cpp stuff
namespace layers {    
   
    Scan::Scan(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/scan.spv");       
        dev = backend::device;
    }
       
        
    void Scan::init( int _body,  int _num_scan_inputs,  std::vector<int> _scan_input_axes,  std::vector<int> _scan_input_directions,  std::vector<int> _scan_output_axes,  std::vector<int> _scan_output_directions) {      
		 body = _body; 
 		 num_scan_inputs = _num_scan_inputs; 
 		 scan_input_axes = _scan_input_axes; 
 		 scan_input_directions = _scan_input_directions; 
 		 scan_output_axes = _scan_output_axes; 
 		 scan_output_directions = _scan_output_directions; 
  

    }
    
    void Scan::bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _y0_o, std::string _y1_o, std::string _y2_o, std::string _y3_o, std::string _y4_o, std::string _y5_o, std::string _y6_o, std::string _y7_o, std::string _y8_o, std::string _y9_o, std::string _y10_o, std::string _y11_o, std::string _y12_o, std::string _y13_o, std::string _y14_o, std::string _y15_o, std::string _y16_o, std::string _y17_o, std::string _y18_o, std::string _y19_o, std::string _y20_o, std::string _y21_o, std::string _y22_o, std::string _y23_o, std::string _y24_o, std::string _y25_o, std::string _y26_o, std::string _y27_o, std::string _y28_o, std::string _y29_o, std::string _y30_o, std::string _y31_o){    
        x0_i = _x0_i; x1_i = _x1_i; x2_i = _x2_i; x3_i = _x3_i; x4_i = _x4_i; x5_i = _x5_i; x6_i = _x6_i; x7_i = _x7_i; x8_i = _x8_i; x9_i = _x9_i; x10_i = _x10_i; x11_i = _x11_i; x12_i = _x12_i; x13_i = _x13_i; x14_i = _x14_i; x15_i = _x15_i; x16_i = _x16_i; x17_i = _x17_i; x18_i = _x18_i; x19_i = _x19_i; x20_i = _x20_i; x21_i = _x21_i; x22_i = _x22_i; x23_i = _x23_i; x24_i = _x24_i; x25_i = _x25_i; x26_i = _x26_i; x27_i = _x27_i; x28_i = _x28_i; x29_i = _x29_i; x30_i = _x30_i; x31_i = _x31_i; y0_o = _y0_o; y1_o = _y1_o; y2_o = _y2_o; y3_o = _y3_o; y4_o = _y4_o; y5_o = _y5_o; y6_o = _y6_o; y7_o = _y7_o; y8_o = _y8_o; y9_o = _y9_o; y10_o = _y10_o; y11_o = _y11_o; y12_o = _y12_o; y13_o = _y13_o; y14_o = _y14_o; y15_o = _y15_o; y16_o = _y16_o; y17_o = _y17_o; y18_o = _y18_o; y19_o = _y19_o; y20_o = _y20_o; y21_o = _y21_o; y22_o = _y22_o; y23_o = _y23_o; y24_o = _y24_o; y25_o = _y25_o; y26_o = _y26_o; y27_o = _y27_o; y28_o = _y28_o; y29_o = _y29_o; y30_o = _y30_o; y31_o = _y31_o;        
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
 
		SHAPES.push_back(backend::tensor_dict[y0_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y1_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y2_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y3_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y4_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y5_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y6_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y7_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y8_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y9_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y10_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y11_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y12_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y13_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y14_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y15_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y16_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y17_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y18_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y19_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y20_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y21_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y22_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y23_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y24_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y25_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y26_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y27_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y28_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y29_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y30_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[y31_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Scan::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[x0_i]->data, *backend::tensor_dict[x1_i]->data, *backend::tensor_dict[x2_i]->data, *backend::tensor_dict[x3_i]->data, *backend::tensor_dict[x4_i]->data, *backend::tensor_dict[x5_i]->data, *backend::tensor_dict[x6_i]->data, *backend::tensor_dict[x7_i]->data, *backend::tensor_dict[x8_i]->data, *backend::tensor_dict[x9_i]->data, *backend::tensor_dict[x10_i]->data, *backend::tensor_dict[x11_i]->data, *backend::tensor_dict[x12_i]->data, *backend::tensor_dict[x13_i]->data, *backend::tensor_dict[x14_i]->data, *backend::tensor_dict[x15_i]->data, *backend::tensor_dict[x16_i]->data, *backend::tensor_dict[x17_i]->data, *backend::tensor_dict[x18_i]->data, *backend::tensor_dict[x19_i]->data, *backend::tensor_dict[x20_i]->data, *backend::tensor_dict[x21_i]->data, *backend::tensor_dict[x22_i]->data, *backend::tensor_dict[x23_i]->data, *backend::tensor_dict[x24_i]->data, *backend::tensor_dict[x25_i]->data, *backend::tensor_dict[x26_i]->data, *backend::tensor_dict[x27_i]->data, *backend::tensor_dict[x28_i]->data, *backend::tensor_dict[x29_i]->data, *backend::tensor_dict[x30_i]->data, *backend::tensor_dict[x31_i]->data, *backend::tensor_dict[y0_o]->data, *backend::tensor_dict[y1_o]->data, *backend::tensor_dict[y2_o]->data, *backend::tensor_dict[y3_o]->data, *backend::tensor_dict[y4_o]->data, *backend::tensor_dict[y5_o]->data, *backend::tensor_dict[y6_o]->data, *backend::tensor_dict[y7_o]->data, *backend::tensor_dict[y8_o]->data, *backend::tensor_dict[y9_o]->data, *backend::tensor_dict[y10_o]->data, *backend::tensor_dict[y11_o]->data, *backend::tensor_dict[y12_o]->data, *backend::tensor_dict[y13_o]->data, *backend::tensor_dict[y14_o]->data, *backend::tensor_dict[y15_o]->data, *backend::tensor_dict[y16_o]->data, *backend::tensor_dict[y17_o]->data, *backend::tensor_dict[y18_o]->data, *backend::tensor_dict[y19_o]->data, *backend::tensor_dict[y20_o]->data, *backend::tensor_dict[y21_o]->data, *backend::tensor_dict[y22_o]->data, *backend::tensor_dict[y23_o]->data, *backend::tensor_dict[y24_o]->data, *backend::tensor_dict[y25_o]->data, *backend::tensor_dict[y26_o]->data, *backend::tensor_dict[y27_o]->data, *backend::tensor_dict[y28_o]->data, *backend::tensor_dict[y29_o]->data, *backend::tensor_dict[y30_o]->data, *backend::tensor_dict[y31_o]->data);
    }

    void Scan::forward(){ 
        program->run();
    }

}

