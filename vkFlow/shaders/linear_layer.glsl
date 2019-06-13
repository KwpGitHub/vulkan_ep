#version 460

layout(local_size_x_id = 0) in;             // workgroup size (set with .spec(64) on C++ side)
layout(push_constant) uniform Parameters {  // push constants (set with {128, 0.1} on C++ side)
   uint size;                               // array size
   float a;                                 // scaling parameter
} params;

layout(std430, binding = 0) buffer lay0 { float arr_y[]; }; // array parameters
layout(std430, binding = 1) buffer lay1 { float arr_x[]; };

void main(){
   const uint id = gl_GlobalInvocationID.x; // current offset
   if(params.size <= id){                   // drop threads outside the buffer
      return;
   }
   arr_y[id] += params.a*arr_x[id];         // saxpy
}