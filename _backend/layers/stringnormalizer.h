#include "../layer.h"
#ifndef STRINGNORMALIZER_H
#define STRINGNORMALIZER_H 
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
        StringNormalizer(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _case_change_action,  int _is_case_sensitive,  int _locale); 
        void bind(std::string _stopwords, std::string _X_input, std::string _Y_output); 

        ~StringNormalizer() {}

    };
    
}

#endif

