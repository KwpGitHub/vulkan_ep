#ifndef QLINEARCONV_H
#define QLINEARCONV_H //QLinearConv
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   x_input, x_scale_input, x_zero_point_input, w_input, w_scale_input, w_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct QLinearConv_parameter_descriptor{    
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
    };   

    struct QLinearConv_input_desriptor{
        Tensor* x_input; Tensor* x_scale_input; Tensor* x_zero_point_input; Tensor* w_input; Tensor* w_scale_input; Tensor* w_zero_point_input; Tensor* y_scale_input; Tensor* y_zero_point_input;
        Tensor* B_input_opt;
    };

    struct QLinearConv_output_descriptor{
        Tensor* y_output;
        
    };

    struct QLinearConv_binding_descriptor{
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        Shape_t x_input; Shape_t x_scale_input; Shape_t x_zero_point_input; Shape_t w_input; Shape_t w_scale_input; Shape_t w_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
        Shape_t B_input_opt;
        Shape_t y_output;
        
    };
}


namespace backend {

    class QLinearConv : public Layer {
        QLinearConv_parameter_descriptor parameters;
        QLinearConv_input_desriptor      input;
        QLinearConv_output_descriptor    output;
        QLinearConv_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, QLinearConv_binding_descriptor>* program;
        
    public:
        QLinearConv(std::string, QLinearConv_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~QLinearConv() {}

    };
}

//cpp stuff
namespace backend {    
   
    QLinearConv::QLinearConv(std::string n, QLinearConv_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, QLinearConv_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearconv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* QLinearConv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<QLinearConv, Layer>(m, "QLinearConv")
            .def("forward", &QLinearConv::forward);    
    }*/
}

#endif
