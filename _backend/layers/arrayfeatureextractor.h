#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H //ArrayFeatureExtractor
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ArrayFeatureExtractor_parameter_descriptor{    
        
    };   

    struct ArrayFeatureExtractor_input_desriptor{
        Tensor* X_input; Tensor* Y_input;
        
    };

    struct ArrayFeatureExtractor_output_descriptor{
        Tensor* Z_output;
        
    };

    struct ArrayFeatureExtractor_binding_descriptor{
        
		
        Shape_t X_input; Shape_t Y_input;
        
        Shape_t Z_output;
        
    };
}


namespace backend {

    class ArrayFeatureExtractor : public Layer {
        ArrayFeatureExtractor_parameter_descriptor parameters;
        ArrayFeatureExtractor_input_desriptor      input;
        ArrayFeatureExtractor_output_descriptor    output;
        ArrayFeatureExtractor_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ArrayFeatureExtractor_binding_descriptor>* program;
        
    public:
        ArrayFeatureExtractor(std::string, ArrayFeatureExtractor_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ArrayFeatureExtractor() {}

    };
}

//cpp stuff
namespace backend {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string n, ArrayFeatureExtractor_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ArrayFeatureExtractor_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/arrayfeatureextractor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ArrayFeatureExtractor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ArrayFeatureExtractor, Layer>(m, "ArrayFeatureExtractor")
            .def("forward", &ArrayFeatureExtractor::forward);    
    }*/
}

#endif
