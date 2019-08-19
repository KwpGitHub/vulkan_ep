#include "Scan.h"
//cpp stuff
namespace backend {    
   
    Scan::Scan() : Layer() { }
       
    vuh::Device* Scan::_get_device() {
        
        return device;
    }
    
    void Scan::init( int _body,  int _num_scan_inputs,  Shape_t _scan_input_axes,  Shape_t _scan_input_directions,  Shape_t _scan_output_axes,  Shape_t _scan_output_directions) {      
		 body = _body; 
 		 num_scan_inputs = _num_scan_inputs; 
 		 scan_input_axes = _scan_input_axes; 
 		 scan_input_directions = _scan_input_directions; 
 		 scan_output_axes = _scan_output_axes; 
 		 scan_output_directions = _scan_output_directions; 
  
    }
    
    void Scan::bind(){
        


		binding.body = body;
  		binding.num_scan_inputs = num_scan_inputs;
  		binding.scan_input_axes = scan_input_axes;
  		binding.scan_input_directions = scan_input_directions;
  		binding.scan_output_axes = scan_output_axes;
  		binding.scan_output_directions = scan_output_directions;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scan.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding);
    }



}



