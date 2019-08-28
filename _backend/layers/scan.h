#ifndef SCAN_H
#define SCAN_H 

#include "../layer.h"

/*

Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ... 
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }


input: Initial values of the loop's N state variables followed by M scan_inputs
output: Final values of the loop's N state variables followed by K scan_outputs
*/

//Scan
//INPUTS:                   
//OPTIONAL_INPUTS:          x0_i, x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i, x9_i, x10_i, x11_i, x12_i, x13_i, x14_i, x15_i, x16_i, x17_i, x18_i, x19_i, x20_i, x21_i, x22_i, x23_i, x24_i, x25_i, x26_i, x27_i, x28_i, x29_i, x30_i, x31_i
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         y0_o, y1_o, y2_o, y3_o, y4_o, y5_o, y6_o, y7_o, y8_o, y9_o, y10_o, y11_o, y12_o, y13_o, y14_o, y15_o, y16_o, y17_o, y18_o, y19_o, y20_o, y21_o, y22_o, y23_o, y24_o, y25_o, y26_o, y27_o, y28_o, y29_o, y30_o, y31_o
//PARAMETERS:               body, num_scan_inputs
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>


//class stuff
namespace layers {   

    class Scan : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int body; int num_scan_inputs; std::vector<int> scan_input_axes; std::vector<int> scan_input_directions; std::vector<int> scan_output_axes; std::vector<int> scan_output_directions;
        
        std::string x0_i; std::string x1_i; std::string x2_i; std::string x3_i; std::string x4_i; std::string x5_i; std::string x6_i; std::string x7_i; std::string x8_i; std::string x9_i; std::string x10_i; std::string x11_i; std::string x12_i; std::string x13_i; std::string x14_i; std::string x15_i; std::string x16_i; std::string x17_i; std::string x18_i; std::string x19_i; std::string x20_i; std::string x21_i; std::string x22_i; std::string x23_i; std::string x24_i; std::string x25_i; std::string x26_i; std::string x27_i; std::string x28_i; std::string x29_i; std::string x30_i; std::string x31_i;
        
        std::string y0_o; std::string y1_o; std::string y2_o; std::string y3_o; std::string y4_o; std::string y5_o; std::string y6_o; std::string y7_o; std::string y8_o; std::string y9_o; std::string y10_o; std::string y11_o; std::string y12_o; std::string y13_o; std::string y14_o; std::string y15_o; std::string y16_o; std::string y17_o; std::string y18_o; std::string y19_o; std::string y20_o; std::string y21_o; std::string y22_o; std::string y23_o; std::string y24_o; std::string y25_o; std::string y26_o; std::string y27_o; std::string y28_o; std::string y29_o; std::string y30_o; std::string y31_o;

        binding_descriptor   binding;
       

    public:
        Scan(std::string name);
        
        virtual void forward();        
        virtual void init( int _body,  int _num_scan_inputs,  std::vector<int> _scan_input_axes,  std::vector<int> _scan_input_directions,  std::vector<int> _scan_output_axes,  std::vector<int> _scan_output_directions); 
        virtual void bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _y0_o, std::string _y1_o, std::string _y2_o, std::string _y3_o, std::string _y4_o, std::string _y5_o, std::string _y6_o, std::string _y7_o, std::string _y8_o, std::string _y9_o, std::string _y10_o, std::string _y11_o, std::string _y12_o, std::string _y13_o, std::string _y14_o, std::string _y15_o, std::string _y16_o, std::string _y17_o, std::string _y18_o, std::string _y19_o, std::string _y20_o, std::string _y21_o, std::string _y22_o, std::string _y23_o, std::string _y24_o, std::string _y25_o, std::string _y26_o, std::string _y27_o, std::string _y28_o, std::string _y29_o, std::string _y30_o, std::string _y31_o); 
        virtual void build();

        ~Scan() {}
    };
   
}
#endif

