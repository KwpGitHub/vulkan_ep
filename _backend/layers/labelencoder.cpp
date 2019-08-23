#include "labelencoder.h"
//cpp stuff
namespace layers {    
   
    LabelEncoder::LabelEncoder(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/labelencoder.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* LabelEncoder::_get_device() {
        
        return backend::device;
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

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.default_float = default_float;
  		//binding.default_int64 = default_int64;
  		//binding.default_string = default_string;
  		//binding.keys_floats = keys_floats;
  		//binding.keys_int64s = keys_int64s;
  		//binding.keys_strings = keys_strings;
  		//binding.values_floats = values_floats;
  		//binding.values_int64s = values_int64s;
  		//binding.values_strings = values_strings;
         
    }

    void LabelEncoder::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

