#include "Shape.h"
//cpp stuff
namespace backend {    
   
    Shape::Shape(const std::string& name) : Layer(name) { }
       
    vuh::Device* Shape::_get_device() {
        
        return device;
    }
    
    void Shape::init() {      
  
    }
    
    void Shape::bind(std::string _data_i, std::string _shape_o){
        data_i = _data_i; shape_o = _shape_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.shape_o = tensor_dict[shape_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shape.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[shape_o]->data());
    }

}

