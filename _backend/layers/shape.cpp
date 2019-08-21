#include "Shape.h"
//cpp stuff
namespace backend {    
   
    Shape::Shape(std::string name) : Layer(name) { }
       
    vuh::Device* Shape::_get_device() {
        
        return device;
    }
    
    void Shape::init() {      
  
    }
    
    void Shape::bind(std::string _data_i, std::string _shape_o){
        data_i = _data_i; shape_o = _shape_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.shape_o = tensor_dict[shape_o]->shape();
 


        
    }
}

