#ifndef DROPOUT_H
#define DROPOUT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: The input data as Tensor.
output: The output.
output: The output mask.

*/
//Dropout
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         mask_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      ratio
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Dropout : public Layer {
        typedef struct {    
            float ratio;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            Tensor* mask_output_opt;
        } output_descriptor;

        typedef struct {
            float ratio;
		
            Shape_t data_input;
            
            Shape_t output_output;
            Shape_t mask_output_opt;
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Dropout(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Dropout() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Dropout::Dropout(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dropout.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Dropout::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Dropout::init() {
		binding.data_input = input.data_input->shape();
 
		binding.output_output = output.output_output->shape();
  		binding.mask_output_opt = output.mask_output_opt->shape();
 
		binding.ratio = parameters.ratio;
 
        program->bind(binding, *input.data_input->data(), *output.output_output->data(), *output.mask_output_opt->data());
    }
    
    void Dropout::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Dropout, Layer>(m, "Dropout")
            .def("forward", &Dropout::forward);    
    }
}*/

#endif
