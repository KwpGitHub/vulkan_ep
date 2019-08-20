#ifndef LOOP_H
#define LOOP_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]
      %keepgoing[BOOL, scalar]
      %b[INT32, scalar]
    ) {
      %my_local = Add(%a, %b)
      %b_out = Sub(%a, %b)
      %keepgoing_out = Greater(%my_local, %b_out)
      %user_defined_vals = Add(%b, %b)
      return %keepgoing_out, %b_out, %user_defined_vals
    }

*Sample equivalent C code*

    {
      // User-defined code (enclosing scope) //
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      // End user-defined code //

      // Implicitly-defined code //
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      // End implicitly-defined code //
      for (int i=0; i < max_trip_count && keepgoing; ++i) {
        // User-defined code (loop body) //
        int my_local = a + b; // Reading values in the enclosing scope is fine
        b = a - b; // writes fine if we specify b as a loop-carried dependency
        keepgoing = my_local > b; // keepgoing is a loop-carried dependency
        user_defined_vals[i] = b + b;
        // End user-defined code //
      }
      // my_local = 123; // Can't do this. my_local was defined in the the body

      // These below values are live-out from the loop and therefore accessible
      b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable a here) are in scope and can
   be referenced in the inputs of the loop.
2) Any variables which you wish to make available in the enclosing scope (i.e.
   the variables b and keepgoing) must be declared as either loop-carried
   dependencies (both at the op inputs and output and at the body net input and
   output) or scan_outputs.
3) Values created in the body cannot be accessed in the enclosing scope.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

input: A maximum trip-count for the loop specified at runtime. Optional. Pass empty string to skip.
input: A boolean termination condition. Optional. Pass empty string to skip.
input: The initial values of any loop-carried dependencies (values that change across loop iterations)
output: Final N loop carried dependency values then K scan_outputs
*/

//Loop
//INPUTS:                   
//OPTIONAL_INPUTS:          M_i, cond_i
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Loop : public Layer {
        typedef struct {
            int body;
			
            
            Shape_t M_i; Shape_t cond_i;
            
            
        } binding_descriptor;

        int body;
        
        std::string M_i; std::string cond_i;
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Loop(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _body); 
        void bind(std::string _M_i, std::string _cond_i); 

        ~Loop() {}
    };

}

#endif

