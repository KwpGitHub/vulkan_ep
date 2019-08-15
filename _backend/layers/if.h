#ifndef IF_H
#define IF_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
If conditional
input: Condition for the if
output: Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same shape and same data type.
//*/
//If
//INPUTS:                   cond_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class If : public Layer {
        typedef struct {
            int else_branch; int then_branch;
			
            Shape_t cond_input;
            
            
            
        } binding_descriptor;

        int else_branch; int then_branch;
        std::string cond_input;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        If(std::string n, int else_branch, int then_branch);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string cond_input); 

        ~If() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    If::If(std::string n, int else_branch, int then_branch) : Layer(n) { }
       
    vuh::Device* If::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void If::init() {      
    
		binding.cond_input = tensor_dict[cond_input]->shape();
 

		binding.else_branch = else_branch;
  		binding.then_branch = then_branch;
 
    }
    
    void If::call(std::string cond_input){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[cond_input]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<If, Layer>(m, "If")
            .def(py::init<std::string, int, int> ())
            .def("forward", &If::forward)
            .def("init", &If::init)
            .def("call", (void (If::*) (std::string)) &If::call);
    }
}

#endif

/* PYTHON STUFF

*/

