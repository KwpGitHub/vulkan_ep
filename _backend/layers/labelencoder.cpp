#include "labelencoder.h"
//cpp stuff
namespace layers {    
   
    LabelEncoder::LabelEncoder(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/labelencoder.spv");       
        dev = backend::device;
    }
       
        
    void LabelEncoder::init( float _default_float,  int _default_int64,  std::string _default_string,  std::vector<float> _keys_floats,  std::vector<int> _keys_int64s,  std::vector<std::string> _keys_strings,  std::vector<float> _values_floats,  std::vector<int> _values_int64s,  std::vector<std::string> _values_strings) {      
		 default_float = _default_float; 
 		 default_int64 = _default_int64; 
 		 default_string = _default_string; 
 		 keys_floats = _keys_floats; 
 		 keys_int64s = _keys_int64s; 
 		 keys_strings = _keys_strings; 
 		 values_floats = _values_floats; 
 		 values_int64s = _values_int64s; 
 		 values_strings = _values_strings; 
  

    }
    
    void LabelEncoder::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LabelEncoder::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void LabelEncoder::forward(){ 
        program->run();
    }

}

