#ifndef STRINGNORMALIZER_H
#define STRINGNORMALIZER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].

input: UTF-8 strings to normalize
output: UTF-8 Normalized strings
//*/
//StringNormalizer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      case_change_action, is_case_sensitive, locale, stopwords
//OPTIONAL_PARAMETERS_TYPE: int, int, int, Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class StringNormalizer : public Layer {
        typedef struct {
            int case_change_action; int is_case_sensitive; int locale;
			Shape_t stopwords;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int case_change_action; int is_case_sensitive; int locale; std::string stopwords;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        StringNormalizer(std::string n, int case_change_action, int is_case_sensitive, int locale);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string stopwords, std::string X_input, std::string Y_output); 

        ~StringNormalizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(std::string n, int case_change_action, int is_case_sensitive, int locale) : Layer(n) { }
       
    vuh::Device* StringNormalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void StringNormalizer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.case_change_action = case_change_action;
  		binding.is_case_sensitive = is_case_sensitive;
  		binding.locale = locale;
  		binding.stopwords = tensor_dict[stopwords]->shape();
 
    }
    
    void StringNormalizer::call(std::string stopwords, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[stopwords]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<StringNormalizer, Layer>(m, "StringNormalizer")
            .def(py::init<std::string, int, int, int> ())
            .def("forward", &StringNormalizer::forward)
            .def("init", &StringNormalizer::init)
            .def("call", (void (StringNormalizer::*) (std::string, std::string, std::string)) &StringNormalizer::call);
    }
}

#endif

/* PYTHON STUFF

*/

