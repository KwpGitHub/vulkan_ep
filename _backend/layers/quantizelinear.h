#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.

input: N-D full precision Input tensor to be quantized.
input: Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization. Default value is 0 if it's not specified.
output: N-D quantized output tensor. It has same shape as input 'x'.

*/
//QuantizeLinear
//INPUTS:                   x_input, y_scale_input
//OPTIONAL_INPUTS:          y_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class QuantizeLinear : public Layer {
        typedef struct {    
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* x_input; Tensor* y_scale_input;
            Tensor* y_zero_point_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* y_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t x_input; Shape_t y_scale_input;
            Shape_t y_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QuantizeLinear(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~QuantizeLinear() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* QuantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QuantizeLinear::init() {
		binding.x_input = input.x_input->shape();
  		binding.y_scale_input = input.y_scale_input->shape();
  		binding.y_zero_point_input_opt = input.y_zero_point_input_opt->shape();
 
		binding.y_output = output.y_output->shape();
 

        program->bind(binding, *input.x_input->data(), *input.y_scale_input->data(), *input.y_zero_point_input_opt->data(), *output.y_output->data());
    }
    
    void QuantizeLinear::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<QuantizeLinear, Layer>(m, "QuantizeLinear")
            .def("forward", &QuantizeLinear::forward);    
    }
}*/

#endif
