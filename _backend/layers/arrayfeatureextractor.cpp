#include "ArrayFeatureExtractor.h"

//cpp stuff
namespace backend {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string n) : Layer(n) { }
       
    vuh::Device* ArrayFeatureExtractor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArrayFeatureExtractor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 

    }
    
    void ArrayFeatureExtractor::call(std::string X_input, std::string Y_input, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/arrayfeatureextractor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[Z_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


