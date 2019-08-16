#include "Loop.h"

//cpp stuff
namespace backend {    
   
    Loop::Loop(std::string n) : Layer(n) { }
       
    vuh::Device* Loop::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Loop::init( int _body) {      
		 body = _body; 
  
    }
    
    void Loop::bind(std::string _M_input_opt, std::string _cond_input_opt){
        M_input_opt = _M_input_opt; cond_input_opt = _cond_input_opt;
		binding.M_input_opt = tensor_dict[M_input_opt]->shape();
  		binding.cond_input_opt = tensor_dict[cond_input_opt]->shape();
 

		binding.body = body;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/loop.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[M_input_opt]->data(), *tensor_dict[cond_input_opt]->data());
    }
    
}

    //backend::nn;

//python stuff


