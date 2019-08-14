#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H //FeatureVectorizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      inputdimensions
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct FeatureVectorizer_parameter_descriptor{    
        Shape_t inputdimensions;
    };   

    struct FeatureVectorizer_input_desriptor{
        
        
    };

    struct FeatureVectorizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct FeatureVectorizer_binding_descriptor{
        Shape_t inputdimensions;
		
        
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class FeatureVectorizer : public Layer {
        FeatureVectorizer_parameter_descriptor parameters;
        FeatureVectorizer_input_desriptor      input;
        FeatureVectorizer_output_descriptor    output;
        FeatureVectorizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, FeatureVectorizer_binding_descriptor>* program;
        
    public:
        FeatureVectorizer(std::string, FeatureVectorizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~FeatureVectorizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    FeatureVectorizer::FeatureVectorizer(std::string n, FeatureVectorizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, FeatureVectorizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/featurevectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* FeatureVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<FeatureVectorizer, Layer>(m, "FeatureVectorizer")
            .def("forward", &FeatureVectorizer::forward);    
    }*/
}

#endif
