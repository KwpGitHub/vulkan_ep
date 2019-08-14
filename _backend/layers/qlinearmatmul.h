#ifndef QLINEARMATMUL_H
#define QLINEARMATMUL_H //QLinearMatMul
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   a_input, a_scale_input, a_zero_point_input, b_input, b_scale_input, b_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct QLinearMatMul_parameter_descriptor{    
        
    };   

    struct QLinearMatMul_input_desriptor{
        Tensor* a_input; Tensor* a_scale_input; Tensor* a_zero_point_input; Tensor* b_input; Tensor* b_scale_input; Tensor* b_zero_point_input; Tensor* y_scale_input; Tensor* y_zero_point_input;
        
    };

    struct QLinearMatMul_output_descriptor{
        Tensor* y_output;
        
    };

    struct QLinearMatMul_binding_descriptor{
        
		
        Shape_t a_input; Shape_t a_scale_input; Shape_t a_zero_point_input; Shape_t b_input; Shape_t b_scale_input; Shape_t b_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
        
        Shape_t y_output;
        
    };
}


namespace backend {

    class QLinearMatMul : public Layer {
        QLinearMatMul_parameter_descriptor parameters;
        QLinearMatMul_input_desriptor      input;
        QLinearMatMul_output_descriptor    output;
        QLinearMatMul_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, QLinearMatMul_binding_descriptor>* program;
        
    public:
        QLinearMatMul(std::string, QLinearMatMul_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~QLinearMatMul() {}

    };
}

//cpp stuff
namespace backend {    
   
    QLinearMatMul::QLinearMatMul(std::string n, QLinearMatMul_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, QLinearMatMul_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* QLinearMatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<QLinearMatMul, Layer>(m, "QLinearMatMul")
            .def("forward", &QLinearMatMul::forward);    
    }*/
}

#endif
