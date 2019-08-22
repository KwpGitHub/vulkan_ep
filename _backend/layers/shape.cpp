#include "shape.h"
//cpp stuff
namespace layers {    
   
    Shape::Shape(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\shape.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Shape::_get_device() {
        
        return backend::device;
    }
    
    void Shape::init() {      
  
    }
    
    void Shape::bind(std::string _data_i, std::string _shape_o){
        data_i = _data_i; shape_o = _shape_o;

		//binding.data_i = tensor_dict[data_i]->shape();
 
		//binding.shape_o = tensor_dict[shape_o]->shape();
 
        
    }

    void Shape::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[shape_o]->data());
    }

}

