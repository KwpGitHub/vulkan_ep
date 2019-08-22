#include "scaler.h"
//cpp stuff
namespace layers {    
   
    Scaler::Scaler(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\scaler.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Scaler::_get_device() {
        
        return backend::device;
    }
    
    void Scaler::init( std::vector<float> _offset,  std::vector<float> _scale) {      
		 offset = _offset; 
 		 scale = _scale; 
  
    }
    
    void Scaler::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.offset = offset;
  		//binding.scale = scale;
         
    }

    void Scaler::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

