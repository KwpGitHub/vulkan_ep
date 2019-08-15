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
//*/
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
            
			
            Shape_t x_input; Shape_t y_scale_input;
            Shape_t y_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        
        std::string x_input; std::string y_scale_input;
        std::string y_zero_point_input_opt;
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QuantizeLinear(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string x_input, std::string y_scale_input, std::string y_zero_point_input_opt, std::string y_output); 

        ~QuantizeLinear() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(std::string n) : Layer(n) { }
       
    vuh::Device* QuantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QuantizeLinear::init() {      
    
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input_opt = tensor_dict[y_zero_point_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 

    }
    
    void QuantizeLinear::call(std::string x_input, std::string y_scale_input, std::string y_zero_point_input_opt, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input_opt]->data(), *tensor_dict[y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<QuantizeLinear, Layer>(m, "QuantizeLinear")
            .def(py::init<std::string> ())
            .def("forward", &QuantizeLinear::forward)
            .def("init", &QuantizeLinear::init)
            .def("call", (void (QuantizeLinear::*) (std::string, std::string, std::string, std::string)) &QuantizeLinear::call);
    }
}

#endif

/* PYTHON STUFF

*/

