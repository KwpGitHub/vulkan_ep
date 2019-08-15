#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.


input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
input: The input 1-dimensional scale tensor of size C.
input: The input 1-dimensional bias tensor of size C.
output: The output tensor of the same shape as input.

*/
//InstanceNormalization
//INPUTS:                   input_input, scale_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//class stuff
namespace backend {   

    class InstanceNormalization : public Layer {
        typedef struct {    
            float epsilon;
        } parameter_descriptor;  

        typedef struct {
            Tensor* input_input; Tensor* scale_input; Tensor* B_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            float epsilon;
		
            Shape_t input_input; Shape_t scale_input; Shape_t B_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        InstanceNormalization(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~InstanceNormalization() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    InstanceNormalization::InstanceNormalization(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/instancenormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* InstanceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void InstanceNormalization::init() {
		binding.input_input = input.input_input->shape();
  		binding.scale_input = input.scale_input->shape();
  		binding.B_input = input.B_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.epsilon = parameters.epsilon;
 
        program->bind(binding, *input.input_input->data(), *input.scale_input->data(), *input.B_input->data(), *output.output_output->data());
    }
    
    void InstanceNormalization::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<InstanceNormalization, Layer>(m, "InstanceNormalization")
            .def("forward", &InstanceNormalization::forward);    
    }
}*/

#endif
