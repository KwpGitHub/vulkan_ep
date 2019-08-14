#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H //LinearClassifier
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LinearClassifier_parameter_descriptor{    
        Tensor* coefficients; Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* intercepts; int multi_class; int post_transform;
    };   

    struct LinearClassifier_input_desriptor{
        Tensor* X_input;
        
    };

    struct LinearClassifier_output_descriptor{
        Tensor* Y_output; Tensor* Z_output;
        
    };

    struct LinearClassifier_binding_descriptor{
        Shape_t classlabels_ints; int multi_class; int post_transform;
		Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
        Shape_t X_input;
        
        Shape_t Y_output; Shape_t Z_output;
        
    };
}


namespace backend {

    class LinearClassifier : public Layer {
        LinearClassifier_parameter_descriptor parameters;
        LinearClassifier_input_desriptor      input;
        LinearClassifier_output_descriptor    output;
        LinearClassifier_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LinearClassifier_binding_descriptor>* program;
        
    public:
        LinearClassifier(std::string, LinearClassifier_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LinearClassifier() {}

    };
}

//cpp stuff
namespace backend {    
   
    LinearClassifier::LinearClassifier(std::string n, LinearClassifier_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LinearClassifier_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LinearClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LinearClassifier, Layer>(m, "LinearClassifier")
            .def("forward", &LinearClassifier::forward);    
    }*/
}

#endif
