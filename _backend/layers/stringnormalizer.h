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

*/
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
            int case_change_action; int is_case_sensitive; int locale; Tensor* stopwords;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int case_change_action; int is_case_sensitive; int locale;
		Shape_t stopwords;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        StringNormalizer(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~StringNormalizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* StringNormalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void StringNormalizer::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.case_change_action = parameters.case_change_action;
  		binding.is_case_sensitive = parameters.is_case_sensitive;
  		binding.locale = parameters.locale;
  		binding.stopwords = parameters.stopwords->shape();
 
        program->bind(binding, *parameters.stopwords->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void StringNormalizer::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<StringNormalizer, Layer>(m, "StringNormalizer")
            .def("forward", &StringNormalizer::forward);    
    }
}*/

#endif
