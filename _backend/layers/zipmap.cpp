#include "ZipMap.h"

//cpp stuff
namespace backend {    
   
    ZipMap::ZipMap(std::string n) : Layer(n) { }
       
    vuh::Device* ZipMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ZipMap::init( Shape_t _classlabels_int64s) {      
		 classlabels_int64s = _classlabels_int64s; 
  
    }
    
    void ZipMap::bind(std::string _classlabels_strings, std::string _X_input, std::string _Z_output){
        classlabels_strings = _classlabels_strings; X_input = _X_input; Z_output = _Z_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.classlabels_int64s = classlabels_int64s;
 
		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Z_output]->data());
    }
    
}

    //backend::nn;

//python stuff


