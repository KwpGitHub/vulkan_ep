#ifndef DEQUANTIZELINEAR_H
#define DEQUANTIZELINEAR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).

input: N-D quantized input tensor to be de-quantized.
input: Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. It's optional. 0 is the default value when it's not specified.
output: N-D full precision output tensor. It has same shape as input 'x'.

*/
//DequantizeLinear
//INPUTS:                   x_input, x_scale_input
//OPTIONAL_INPUTS:          x_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class DequantizeLinear : public Layer {
        typedef struct {    
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* x_input; Tensor* x_scale_input;
            Tensor* x_zero_point_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* y_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t x_input; Shape_t x_scale_input;
            Shape_t x_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DequantizeLinear(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~DequantizeLinear() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    DequantizeLinear::DequantizeLinear(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dequantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* DequantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void DequantizeLinear::init() {
		binding.x_input = input.x_input->shape();
  		binding.x_scale_input = input.x_scale_input->shape();
  		binding.x_zero_point_input_opt = input.x_zero_point_input_opt->shape();
 
		binding.y_output = output.y_output->shape();
 

        program->bind(binding, *input.x_input->data(), *input.x_scale_input->data(), *input.x_zero_point_input_opt->data(), *output.y_output->data());
    }
    
    void DequantizeLinear::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<DequantizeLinear, Layer>(m, "DequantizeLinear")
            .def("forward", &DequantizeLinear::forward);    
    }
}*/

#endif
