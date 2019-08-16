#include "If.h"

//cpp stuff
namespace backend {    
   
    If::If(std::string n) : Layer(n) { }
       
    vuh::Device* If::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void If::init( int _else_branch,  int _then_branch) {      
		 else_branch = _else_branch; 
 		 then_branch = _then_branch; 
  
    }
    
    void If::bind(std::string _cond_input){
        cond_input = _cond_input;
		binding.cond_input = tensor_dict[cond_input]->shape();
 

		binding.else_branch = else_branch;
  		binding.then_branch = then_branch;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[cond_input]->data());
    }
    
}

    //backend::nn;

//python stuff


