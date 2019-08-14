#ifndef LABELENCODER_H
#define LABELENCODER_H //LabelEncoder
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      default_float, default_int64, default_string, keys_floats, keys_int64s, keys_strings, values_floats, values_int64s, values_strings
//OPTIONAL_PARAMETERS_TYPE: float, int, int, Tensor*, Shape_t, Tensor*, Tensor*, Shape_t, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LabelEncoder_parameter_descriptor{    
        float default_float; int default_int64; int default_string; Tensor* keys_floats; Shape_t keys_int64s; Tensor* keys_strings; Tensor* values_floats; Shape_t values_int64s; Tensor* values_strings;
    };   

    struct LabelEncoder_input_desriptor{
        Tensor* X_input;
        
    };

    struct LabelEncoder_output_descriptor{
        Tensor* Y_output;
        
    };

    struct LabelEncoder_binding_descriptor{
        float default_float; int default_int64; int default_string; Shape_t keys_int64s; Shape_t values_int64s;
		Shape_t keys_floats; Shape_t keys_strings; Shape_t values_floats; Shape_t values_strings;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class LabelEncoder : public Layer {
        LabelEncoder_parameter_descriptor parameters;
        LabelEncoder_input_desriptor      input;
        LabelEncoder_output_descriptor    output;
        LabelEncoder_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LabelEncoder_binding_descriptor>* program;
        
    public:
        LabelEncoder(std::string, LabelEncoder_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LabelEncoder() {}

    };
}

//cpp stuff
namespace backend {    
   
    LabelEncoder::LabelEncoder(std::string n, LabelEncoder_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LabelEncoder_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/labelencoder.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LabelEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LabelEncoder, Layer>(m, "LabelEncoder")
            .def("forward", &LabelEncoder::forward);    
    }*/
}

#endif
