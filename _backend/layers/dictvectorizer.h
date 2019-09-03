#pragma once
#ifndef DICTVECTORIZER_H
#define DICTVECTORIZER_H 

#include "../layer.h"

/*

    Uses an index mapping to convert a dictionary to an array.<br>
    Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    the key type. The index into the vocabulary array at which the key is found is then
    used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
    The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    Therefore, the output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    Any keys not present in the input dictionary, will be zero in the output array.<br>
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    
input: A dictionary.
output: A 1-D tensor holding values from the input dictionary.
*/

//DictVectorizer
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      int64_vocabulary, string_vocabulary
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>


//class stuff
namespace layers {   

    class DictVectorizer : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> m_int64_vocabulary; std::vector<std::string> m_string_vocabulary;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        DictVectorizer(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _int64_vocabulary,  std::vector<std::string> _string_vocabulary); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~DictVectorizer() {}
    };
   
}
#endif

