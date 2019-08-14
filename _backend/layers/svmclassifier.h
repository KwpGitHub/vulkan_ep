#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H //SVMClassifier
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, Tensor*, int, int, Tensor*, Tensor*, Tensor*, Tensor*, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct SVMClassifier_parameter_descriptor{    
        Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* coefficients; Tensor* kernel_params; int kernel_type; int post_transform; Tensor* prob_a; Tensor* prob_b; Tensor* rho; Tensor* support_vectors; Shape_t vectors_per_class;
    };   

    struct SVMClassifier_input_desriptor{
        Tensor* X_input;
        
    };

    struct SVMClassifier_output_descriptor{
        Tensor* Y_output; Tensor* Z_output;
        
    };

    struct SVMClassifier_binding_descriptor{
        Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class;
		Shape_t classlabels_strings; Shape_t coefficients; Shape_t kernel_params; Shape_t prob_a; Shape_t prob_b; Shape_t rho; Shape_t support_vectors;
        Shape_t X_input;
        
        Shape_t Y_output; Shape_t Z_output;
        
    };
}


namespace backend {

    class SVMClassifier : public Layer {
        SVMClassifier_parameter_descriptor parameters;
        SVMClassifier_input_desriptor      input;
        SVMClassifier_output_descriptor    output;
        SVMClassifier_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, SVMClassifier_binding_descriptor>* program;
        
    public:
        SVMClassifier(std::string, SVMClassifier_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~SVMClassifier() {}

    };
}

//cpp stuff
namespace backend {    
   
    SVMClassifier::SVMClassifier(std::string n, SVMClassifier_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, SVMClassifier_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* SVMClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<SVMClassifier, Layer>(m, "SVMClassifier")
            .def("forward", &SVMClassifier::forward);    
    }*/
}

#endif
