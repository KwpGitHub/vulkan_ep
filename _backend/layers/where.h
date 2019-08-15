#ifndef WHERE_H
#define WHERE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html

input: When True (nonzero), yield X, otherwise yield Y
input: values selected at indices where condition is True
input: values selected at indices where condition is False
output: Tensor of shape equal to the broadcasted shape of condition, X, and Y.
//*/
//Where
//INPUTS:                   condition_input, X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Where : public Layer {
        typedef struct {
            
			
            Shape_t condition_input; Shape_t X_input; Shape_t Y_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string condition_input; std::string X_input; std::string Y_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Where(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string condition_input, std::string X_input, std::string Y_input, std::string output_output); 

        ~Where() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Where::Where(std::string n) : Layer(n) { }
       
    vuh::Device* Where::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Where::init() {      
    
		binding.condition_input = tensor_dict[condition_input]->shape();
  		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Where::call(std::string condition_input, std::string X_input, std::string Y_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/where.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[condition_input]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Where, Layer>(m, "Where")
            .def(py::init<std::string> ())
            .def("forward", &Where::forward)
            .def("init", &Where::init)
            .def("call", (void (Where::*) (std::string, std::string, std::string, std::string)) &Where::call);
    }
}

#endif

/* PYTHON STUFF

*/

