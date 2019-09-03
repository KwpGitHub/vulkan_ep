#include "labelencoder.h"
//cpp stuff
namespace layers {    
   
    LabelEncoder::LabelEncoder(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/labelencoder.spv");       
        dev = backend::g_device;
    }
       
        
    void LabelEncoder::init( float _default_float,  int _default_int64,  std::string _default_string,  std::vector<float> _keys_floats,  std::vector<int> _keys_int64s,  std::vector<std::string> _keys_strings,  std::vector<float> _values_floats,  std::vector<int> _values_int64s,  std::vector<std::string> _values_strings) {      
		 m_default_float = _default_float; 
 		 m_default_int64 = _default_int64; 
 		 m_default_string = _default_string; 
 		 m_keys_floats = _keys_floats; 
 		 m_keys_int64s = _keys_int64s; 
 		 m_keys_strings = _keys_strings; 
 		 m_values_floats = _values_floats; 
 		 m_values_int64s = _values_int64s; 
 		 m_values_strings = _values_strings; 
  

    }
    
    void LabelEncoder::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LabelEncoder::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void LabelEncoder::forward(){ 
        program->run();
    }

}

