#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Given a matrix, apply Lp-normalization along the provided axis.

input: Input matrix
output: Matrix after normalization

*/
//LpNormalization
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class LpNormalization : public Layer {
        typedef struct {    
            int axis; int p;
        } parameter_descriptor;  

        typedef struct {
            Tensor* input_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            int axis; int p;
		
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpNormalization(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~LpNormalization() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LpNormalization::LpNormalization(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lpnormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* LpNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LpNormalization::init() {
		binding.input_input = input.input_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.axis = parameters.axis;
  		binding.p = parameters.p;
 
        program->bind(binding, *input.input_input->data(), *output.output_output->data());
    }
    
    void LpNormalization::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LpNormalization, Layer>(m, "LpNormalization")
            .def("forward", &LpNormalization::forward);    
    }
}*/

#endif
