#include "Shape.h"

//cpp stuff
namespace backend {    
   
    Shape::Shape(std::string n) : Layer(n) { }
       
    vuh::Device* Shape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Shape::init() {      
  
    }
    
    void Shape::bind(std::string _data_input, std::string _shape_output){
        data_input = _data_input; shape_output = _shape_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.shape_output = tensor_dict[shape_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shape.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[shape_output]->data());
    }
    
}

    //backend::nn;

//python stuff


