#include "zipmap.h"
//cpp stuff
namespace layers {    
   
    ZipMap::ZipMap(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\zipmap.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* ZipMap::_get_device() {
        
        return backend::device;
    }
    
    void ZipMap::init( std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings) {      
		 classlabels_int64s = _classlabels_int64s; 
 		 classlabels_strings = _classlabels_strings; 
  
    }
    
    void ZipMap::bind(std::string _X_i, std::string _Z_o){
        X_i = _X_i; Z_o = _Z_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Z_o = tensor_dict[Z_o]->shape();
 
		//binding.classlabels_int64s = classlabels_int64s;
  		//binding.classlabels_strings = classlabels_strings;
         
    }

    void ZipMap::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Z_o]->data());
    }

}

