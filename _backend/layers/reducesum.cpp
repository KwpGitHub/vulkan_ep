#include "reducesum.h"
//cpp stuff
namespace layers {    
   
    ReduceSum::ReduceSum(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/reducesum.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ReduceSum::_get_device() {        
        return backend::device;
    }
    
    void ReduceSum::init( std::vector<int> _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceSum::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = backend::tensor_dict[data_i]->shape();
 
		binding.reduced_o = backend::tensor_dict[reduced_o]->shape();
 
		//binding.axes = axes;
  		//binding.keepdims = keepdims;
         
    }

    void ReduceSum::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[data_i]->data(), *backend::tensor_dict[reduced_o]->data());
    }

    void ReduceSum::forward(){ 
        //program->run();
    }

}

