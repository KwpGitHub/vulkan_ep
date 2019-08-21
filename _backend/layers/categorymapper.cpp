#include "CategoryMapper.h"
//cpp stuff
namespace backend {    
   
    CategoryMapper::CategoryMapper(std::string name) : Layer(name) { }
       
    vuh::Device* CategoryMapper::_get_device() {
        
        return device;
    }
    
    void CategoryMapper::init( Shape_t _cats_int64s,  int _default_int64,  int _default_string) {      
		 cats_int64s = _cats_int64s; 
 		 default_int64 = _default_int64; 
 		 default_string = _default_string; 
  
    }
    
    void CategoryMapper::bind(std::string _cats_strings, std::string _X_i, std::string _Y_o){
        cats_strings = _cats_strings; X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.cats_int64s = cats_int64s;
  		binding.default_int64 = default_int64;
  		binding.default_string = default_string;
 
		binding.cats_strings = tensor_dict[cats_strings]->shape();
 
        
    }
}

