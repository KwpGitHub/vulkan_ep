#ifndef DICTVECTORIZER_H
#define DICTVECTORIZER_H 
#include <pybind11/pybind11.h>
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
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      int64_vocabulary, string_vocabulary
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class DictVectorizer : public Layer {
        typedef struct {    
            Shape_t int64_vocabulary; Tensor* string_vocabulary;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t int64_vocabulary;
		Shape_t string_vocabulary;
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
        DictVectorizer(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~DictVectorizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    DictVectorizer::DictVectorizer(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dictvectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* DictVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void DictVectorizer::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.int64_vocabulary = parameters.int64_vocabulary;
  		binding.string_vocabulary = parameters.string_vocabulary->shape();
 
        program->bind(binding, *parameters.string_vocabulary->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void DictVectorizer::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<DictVectorizer, Layer>(m, "DictVectorizer")
            .def("forward", &DictVectorizer::forward);    
    }
}*/

#endif
