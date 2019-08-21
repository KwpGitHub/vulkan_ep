#ifndef ZIPMAP_H
#define ZIPMAP_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

input: The input values
output: The output map
*/

//ZipMap
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*


//class stuff
namespace backend {   

    class ZipMap : public Layer {
        typedef struct {
            Shape_t classlabels_int64s;
			Shape_t classlabels_strings;
            Shape_t X_i;
            
            Shape_t Z_o;
            
        } binding_descriptor;

        Shape_t classlabels_int64s; std::string classlabels_strings;
        std::string X_i;
        
        std::string Z_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ZipMap(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _classlabels_int64s); 
        virtual void bind(std::string _classlabels_strings, std::string _X_i, std::string _Z_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Z_o]->data());
        }

        ~ZipMap() {}
    };
   
}
#endif

