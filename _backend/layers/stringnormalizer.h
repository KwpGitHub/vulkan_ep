#ifndef STRINGNORMALIZER_H
#define STRINGNORMALIZER_H //StringNormalizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      case_change_action, is_case_sensitive, locale, stopwords
//OPTIONAL_PARAMETERS_TYPE: int, int, int, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct StringNormalizer_parameter_descriptor{    
        int case_change_action; int is_case_sensitive; int locale; Tensor* stopwords;
    };   

    struct StringNormalizer_input_desriptor{
        Tensor* X_input;
        
    };

    struct StringNormalizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct StringNormalizer_binding_descriptor{
        int case_change_action; int is_case_sensitive; int locale;
		Shape_t stopwords;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class StringNormalizer : public Layer {
        StringNormalizer_parameter_descriptor parameters;
        StringNormalizer_input_desriptor      input;
        StringNormalizer_output_descriptor    output;
        StringNormalizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, StringNormalizer_binding_descriptor>* program;
        
    public:
        StringNormalizer(std::string, StringNormalizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~StringNormalizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(std::string n, StringNormalizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, StringNormalizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* StringNormalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<StringNormalizer, Layer>(m, "StringNormalizer")
            .def("forward", &StringNormalizer::forward);    
    }*/
}

#endif
