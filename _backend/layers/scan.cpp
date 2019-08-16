#include "Scan.h"

//cpp stuff
namespace backend {    
   
    Scan::Scan(std::string n, int body, int num_scan_inputs, Shape_t scan_input_axes, Shape_t scan_input_directions, Shape_t scan_output_axes, Shape_t scan_output_directions) : Layer(n) { }
       
    vuh::Device* Scan::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Scan::init() {      
    


		binding.body = body;
  		binding.num_scan_inputs = num_scan_inputs;
  		binding.scan_input_axes = scan_input_axes;
  		binding.scan_input_directions = scan_input_directions;
  		binding.scan_output_axes = scan_output_axes;
  		binding.scan_output_directions = scan_output_directions;
 
    }
    
    void Scan::call(){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scan.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding);
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


