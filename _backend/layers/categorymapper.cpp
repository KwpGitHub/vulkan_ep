#include "CategoryMapper.h"

//cpp stuff
namespace backend {    
   
    CategoryMapper::CategoryMapper(std::string n, Shape_t cats_int64s, int default_int64, int default_string) : Layer(n) { }
       
    vuh::Device* CategoryMapper::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void CategoryMapper::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.cats_int64s = cats_int64s;
  		binding.default_int64 = default_int64;
  		binding.default_string = default_string;
  		binding.cats_strings = tensor_dict[cats_strings]->shape();
 
    }
    
    void CategoryMapper::call(std::string cats_strings, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/categorymapper.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[cats_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


