#ifndef CONSTANT_H
#define CONSTANT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
A constant tensor.

output: Output tensor containing the same value of the provided tensor.

*/
//Constant
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Constant : public Layer {
        typedef struct {    
            Tensor* value;
        } parameter_descriptor;  

        typedef struct {
            
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            
		Shape_t value;
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Constant(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Constant() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Constant::Constant(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Constant::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Constant::init() {

		binding.output_output = output.output_output->shape();
 
		binding.value = parameters.value->shape();
 
        program->bind(binding, *parameters.value->data(), *output.output_output->data());
    }
    
    void Constant::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Constant, Layer>(m, "Constant")
            .def("forward", &Constant::forward);    
    }
}*/

#endif
