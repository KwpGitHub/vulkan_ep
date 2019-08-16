#include "ZipMap.h"

//cpp stuff
namespace backend {    
   
    ZipMap::ZipMap(std::string n, Shape_t classlabels_int64s) : Layer(n) { }
       
    vuh::Device* ZipMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ZipMap::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.classlabels_int64s = classlabels_int64s;
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
 
    }
    
    void ZipMap::call(std::string classlabels_strings, std::string X_input, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Z_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


