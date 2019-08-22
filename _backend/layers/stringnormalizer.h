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
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        std::string case_change_action; int is_case_sensitive; std::string locale; std::vector<std::string> stopwords;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        StringNormalizer(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::string _case_change_action,  int _is_case_sensitive,  std::string _locale,  std::vector<std::string> _stopwords); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~StringNormalizer() {}
    };
   
}
#endif

