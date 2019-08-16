#include "Imputer.h"

//cpp stuff
namespace backend {    
   
    Imputer::Imputer(std::string n, Shape_t imputed_value_int64s, float replaced_value_float, int replaced_value_int64) : Layer(n) { }
       
    vuh::Device* Imputer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Imputer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.imputed_value_int64s = imputed_value_int64s;
  		binding.replaced_value_float = replaced_value_float;
  		binding.replaced_value_int64 = replaced_value_int64;
  		binding.imputed_value_floats = tensor_dict[imputed_value_floats]->shape();
 
    }
    
    void Imputer::call(std::string imputed_value_floats, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/imputer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[imputed_value_floats]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


