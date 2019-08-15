#ifndef MOD_H
#define MOD_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

  Performs element-wise binary modulus (with Numpy-style broadcasting support). 
    The sign of the remainder is the same as that of the Divisor.
  
    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend 
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
    This attribute is set to 0 by default causing the behavior to be like integer mod. 
    Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then `fmod` attribute must be set to 1.
  
    In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: Dividend tensor
input: Divisor tensor
output: Remainder tensor
//*/
//Mod
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      fmod
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Mod : public Layer {
        typedef struct {
            int fmod;
			
            Shape_t A_input; Shape_t B_input;
            
            Shape_t C_output;
            
        } binding_descriptor;

        int fmod;
        std::string A_input; std::string B_input;
        
        std::string C_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mod(std::string n, int fmod);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string A_input, std::string B_input, std::string C_output); 

        ~Mod() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Mod::Mod(std::string n, int fmod) : Layer(n) { }
       
    vuh::Device* Mod::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Mod::init() {      
    
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
 
		binding.C_output = tensor_dict[C_output]->shape();
 
		binding.fmod = fmod;
 
    }
    
    void Mod::call(std::string A_input, std::string B_input, std::string C_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mod.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[C_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Mod, Layer>(m, "Mod")
            .def(py::init<std::string, int> ())
            .def("forward", &Mod::forward)
            .def("init", &Mod::init)
            .def("call", (void (Mod::*) (std::string, std::string, std::string)) &Mod::call);
    }
}

#endif

/* PYTHON STUFF

*/

