#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.

input: An ordered collection of tensors, all with the same element type.
output: The output array, elements ordered as the inputs.

*/
//FeatureVectorizer
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      inputdimensions
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class FeatureVectorizer : public Layer {
        typedef struct {    
            Shape_t inputdimensions;
        } parameter_descriptor;  

        typedef struct {
            
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t inputdimensions;
		
            
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        FeatureVectorizer(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~FeatureVectorizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    FeatureVectorizer::FeatureVectorizer(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/featurevectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* FeatureVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void FeatureVectorizer::init() {

		binding.Y_output = output.Y_output->shape();
 
		binding.inputdimensions = parameters.inputdimensions;
 
        program->bind(binding, *output.Y_output->data());
    }
    
    void FeatureVectorizer::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<FeatureVectorizer, Layer>(m, "FeatureVectorizer")
            .def("forward", &FeatureVectorizer::forward);    
    }
}*/

#endif
