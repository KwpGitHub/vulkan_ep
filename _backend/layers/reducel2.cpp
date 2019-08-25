#include "reducel2.h"
//cpp stuff
namespace layers {    
   
    ReduceL2::ReduceL2(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/reducel2.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ReduceL2::_get_device() {
        
        return backend::device;
    }
    
    void ReduceL2::init( std::vector<int> _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceL2::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		//binding.data_i = tensor_dict[data_i]->shape();
 
		//binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		//binding.axes = axes;
  		//binding.keepdims = keepdims;
         
    }

    void ReduceL2::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
    }

}

