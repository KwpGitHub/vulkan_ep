#pragma once
#ifndef STRINGNORMALIZER_H
#define STRINGNORMALIZER_H 

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
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      case_change_action, is_case_sensitive, locale, stopwords
//OPTIONAL_PARAMETERS_TYPE: std::string, int, std::string, std::vector<std::string>


//class stuff
namespace layers {   

    class StringNormalizer : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string m_case_change_action; int m_is_case_sensitive; std::string m_locale; std::vector<std::string> m_stopwords;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        StringNormalizer(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _case_change_action,  int _is_case_sensitive,  std::string _locale,  std::vector<std::string> _stopwords); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~StringNormalizer() {}
    };
   
}
#endif

