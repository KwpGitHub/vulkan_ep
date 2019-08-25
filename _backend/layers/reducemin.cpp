#include "reducemin.h"
//cpp stuff
namespace layers {    
   
    ReduceMin::ReduceMin(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/reducemin.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ReduceMin::_get_device() {
        
        return backend::device;
    }
    
    void ReduceMin::init( std::vector<int> _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceMin::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		//binding.data_i = tensor_dict[data_i]->shape();
 
		//binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		//binding.axes = axes;
  		//binding.keepdims = keepdims;
         
    }

    void ReduceMin::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
    }

}

