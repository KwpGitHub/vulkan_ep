#ifndef MEANVARIANCENORMALIZATION_H
#define MEANVARIANCENORMALIZATION_H //MeanVarianceNormalization
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MeanVarianceNormalization_parameter_descriptor{    
        Shape_t axes;
    };   

    struct MeanVarianceNormalization_input_desriptor{
        Tensor* X_input;
        
    };

    struct MeanVarianceNormalization_output_descriptor{
        Tensor* Y_output;
        
    };

    struct MeanVarianceNormalization_binding_descriptor{
        Shape_t axes;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class MeanVarianceNormalization : public Layer {
        MeanVarianceNormalization_parameter_descriptor parameters;
        MeanVarianceNormalization_input_desriptor      input;
        MeanVarianceNormalization_output_descriptor    output;
        MeanVarianceNormalization_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MeanVarianceNormalization_binding_descriptor>* program;
        
    public:
        MeanVarianceNormalization(std::string, MeanVarianceNormalization_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MeanVarianceNormalization() {}

    };
}

//cpp stuff
namespace backend {    
   
    MeanVarianceNormalization::MeanVarianceNormalization(std::string n, MeanVarianceNormalization_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MeanVarianceNormalization_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/meanvariancenormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MeanVarianceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MeanVarianceNormalization, Layer>(m, "MeanVarianceNormalization")
            .def("forward", &MeanVarianceNormalization::forward);    
    }*/
}

#endif
