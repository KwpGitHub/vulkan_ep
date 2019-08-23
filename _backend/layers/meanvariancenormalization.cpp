#include "meanvariancenormalization.h"
//cpp stuff
namespace layers {    
   
    MeanVarianceNormalization::MeanVarianceNormalization(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/meanvariancenormalization.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* MeanVarianceNormalization::_get_device() {
        
        return backend::device;
    }
    
    void MeanVarianceNormalization::init( std::vector<int> _axes) {      
		 axes = _axes; 
  
    }
    
    void MeanVarianceNormalization::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.axes = axes;
         
    }

    void MeanVarianceNormalization::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

