#include "LabelEncoder.h"

//cpp stuff
namespace backend {    
   
    LabelEncoder::LabelEncoder(std::string n, float default_float, int default_int64, int default_string, Shape_t keys_int64s, Shape_t values_int64s) : Layer(n) { }
       
    vuh::Device* LabelEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LabelEncoder::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.default_float = default_float;
  		binding.default_int64 = default_int64;
  		binding.default_string = default_string;
  		binding.keys_int64s = keys_int64s;
  		binding.values_int64s = values_int64s;
  		binding.keys_floats = tensor_dict[keys_floats]->shape();
  		binding.keys_strings = tensor_dict[keys_strings]->shape();
  		binding.values_floats = tensor_dict[values_floats]->shape();
  		binding.values_strings = tensor_dict[values_strings]->shape();
 
    }
    
    void LabelEncoder::call(std::string keys_floats, std::string keys_strings, std::string values_floats, std::string values_strings, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/labelencoder.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[keys_floats]->data(), *tensor_dict[keys_strings]->data(), *tensor_dict[values_floats]->data(), *tensor_dict[values_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


