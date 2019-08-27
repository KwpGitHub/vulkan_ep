#include "categorymapper.h"
//cpp stuff
namespace layers {    
   
    CategoryMapper::CategoryMapper(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/categorymapper.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* CategoryMapper::_get_device() {        
        return backend::device;
    }
    
    void CategoryMapper::init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _default_int64,  std::string _default_string) {      
		 cats_int64s = _cats_int64s; 
 		 cats_strings = _cats_strings; 
 		 default_int64 = _default_int64; 
 		 default_string = _default_string; 
  
    }
    
    void CategoryMapper::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.cats_int64s = cats_int64s;
  		//binding.cats_strings = cats_strings;
  		//binding.default_int64 = default_int64;
  		//binding.default_string = default_string;
         
    }

    void CategoryMapper::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void CategoryMapper::forward(){ 
        //program->run();
    }

}

