#ifndef ISINF_H
#define ISINF_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
Map infinity to true and other values to false.
input: input
output: output

*/
//IsInf
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      detect_negative, detect_positive
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class IsInf : public Layer {
        typedef struct {    
            int detect_negative; int detect_positive;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int detect_negative; int detect_positive;
		
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        IsInf(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~IsInf() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    IsInf::IsInf(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/isinf.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* IsInf::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void IsInf::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.detect_negative = parameters.detect_negative;
  		binding.detect_positive = parameters.detect_positive;
 
        program->bind(binding, *input.X_input->data(), *output.Y_output->data());
    }
    
    void IsInf::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<IsInf, Layer>(m, "IsInf")
            .def("forward", &IsInf::forward);    
    }
}*/

#endif
