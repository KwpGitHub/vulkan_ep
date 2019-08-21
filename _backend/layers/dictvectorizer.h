#ifndef DICTVECTORIZER_H
#define DICTVECTORIZER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*


//class stuff
namespace backend {   

    class DictVectorizer : public Layer {
        typedef struct {
            Shape_t int64_vocabulary;
			Shape_t string_vocabulary;
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        Shape_t int64_vocabulary; std::string string_vocabulary;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DictVectorizer(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _int64_vocabulary); 
        virtual void bind(std::string _string_vocabulary, std::string _X_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dictvectorizer.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[string_vocabulary]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~DictVectorizer() {}
    };
   
}
#endif

