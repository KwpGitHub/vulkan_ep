#ifndef SCAN_H
#define SCAN_H //Scan

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body, num_scan_inputs
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t, Shape_t, Shape_t



namespace backend {
    class Scan : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int body; int num_scan_inputs; Shape_t scan_input_axes; Shape_t scan_input_directions; Shape_t scan_output_axes; Shape_t scan_output_directions;
			
            //input
            
            
            //output
            
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Scan(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int body; int num_scan_inputs; Shape_t scan_input_axes; Shape_t scan_input_directions; Shape_t scan_output_axes; Shape_t scan_output_directions;
		
        //input
        
        
        //output
        
        
        //std::vector<uint32_t> output_shape();
   
        ~Scan(){}
    };
}


namespace backend {    
    Scan::Scan(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/scan.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Scan::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
