#ifndef DICTVECTORIZER_H
#define DICTVECTORIZER_H //DictVectorizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      int64_vocabulary, string_vocabulary
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct DictVectorizer_parameter_descriptor{    
        Shape_t int64_vocabulary; Tensor* string_vocabulary;
    };   

    struct DictVectorizer_input_desriptor{
        Tensor* X_input;
        
    };

    struct DictVectorizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct DictVectorizer_binding_descriptor{
        Shape_t int64_vocabulary;
		Shape_t string_vocabulary;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class DictVectorizer : public Layer {
        DictVectorizer_parameter_descriptor parameters;
        DictVectorizer_input_desriptor      input;
        DictVectorizer_output_descriptor    output;
        DictVectorizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, DictVectorizer_binding_descriptor>* program;
        
    public:
        DictVectorizer(std::string, DictVectorizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~DictVectorizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    DictVectorizer::DictVectorizer(std::string n, DictVectorizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, DictVectorizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dictvectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* DictVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<DictVectorizer, Layer>(m, "DictVectorizer")
            .def("forward", &DictVectorizer::forward);    
    }*/
}

#endif
