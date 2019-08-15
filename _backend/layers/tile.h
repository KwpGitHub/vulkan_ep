#ifndef TILE_H
#define TILE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

input: Input tensor of any shape.
input: 1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.
output: Output tensor of the same dimension and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]
//*/
//Tile
//INPUTS:                   input_input, repeats_input
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

    class Tile : public Layer {
        typedef struct {
            
			
            Shape_t input_input; Shape_t repeats_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input; std::string repeats_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Tile(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string repeats_input, std::string output_output); 

        ~Tile() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Tile::Tile(std::string n) : Layer(n) { }
       
    vuh::Device* Tile::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Tile::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.repeats_input = tensor_dict[repeats_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Tile::call(std::string input_input, std::string repeats_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tile.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[repeats_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Tile, Layer>(m, "Tile")
            .def(py::init<std::string> ())
            .def("forward", &Tile::forward)
            .def("init", &Tile::init)
            .def("call", (void (Tile::*) (std::string, std::string, std::string)) &Tile::call);
    }
}

#endif

/* PYTHON STUFF

*/

