#ifndef TFIDFVECTORIZER_H
#define TFIDFVECTORIZER_H //TfIdfVectorizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes
//PARAMETER_TYPES:          int, int, int, int, Shape_t, Shape_t
//OPTIONAL_PARAMETERS:      pool_int64s, pool_strings, weights
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct TfIdfVectorizer_parameter_descriptor{    
        int max_gram_length; int max_skip_count; int min_gram_length; int mode; Shape_t ngram_counts; Shape_t ngram_indexes; Shape_t pool_int64s; Tensor* pool_strings; Tensor* weights;
    };   

    struct TfIdfVectorizer_input_desriptor{
        Tensor* X_input;
        
    };

    struct TfIdfVectorizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct TfIdfVectorizer_binding_descriptor{
        int max_gram_length; int max_skip_count; int min_gram_length; int mode; Shape_t ngram_counts; Shape_t ngram_indexes; Shape_t pool_int64s;
		Shape_t pool_strings; Shape_t weights;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class TfIdfVectorizer : public Layer {
        TfIdfVectorizer_parameter_descriptor parameters;
        TfIdfVectorizer_input_desriptor      input;
        TfIdfVectorizer_output_descriptor    output;
        TfIdfVectorizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, TfIdfVectorizer_binding_descriptor>* program;
        
    public:
        TfIdfVectorizer(std::string, TfIdfVectorizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~TfIdfVectorizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string n, TfIdfVectorizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, TfIdfVectorizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tfidfvectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* TfIdfVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<TfIdfVectorizer, Layer>(m, "TfIdfVectorizer")
            .def("forward", &TfIdfVectorizer::forward);    
    }*/
}

#endif
