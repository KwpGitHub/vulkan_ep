#ifndef POW_H
#define POW_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
input: First operand, base of the exponent.
input: Second operand, power of the exponent.
output: Output tensor (same size as X)
//*/
//Pow
//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Pow : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t Y_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string Y_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pow(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_input, std::string Z_output); 

        ~Pow() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Pow::Pow(std::string n) : Layer(n) { }
       
    vuh::Device* Pow::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Pow::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 

    }
    
    void Pow::call(std::string X_input, std::string Y_input, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pow.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Pow, Layer>(m, "Pow")
            .def(py::init<std::string> ())
            .def("forward", &Pow::forward)
            .def("init", &Pow::init)
            .def("call", (void (Pow::*) (std::string, std::string, std::string)) &Pow::call);
    }
}

#endif

/* PYTHON STUFF

*/

