import numpy as np
import _backend.nn as nn
layer_map = {}


class LSTM:
    name = None
    X_i = str()
    W_i = str()
    R_i = str()
    B_i = str()
    sequence_lens_i = str()
    initial_h_i = str()
    initial_c_i = str()
    P_i = str()
    Y_o = str()
    Y_h_o = str()
    Y_c_o = str()

    #parameters
    activation_alpha = list()
    activation_beta = list()
    activations = list()
    clip = float()
    direction = str()
    hidden_size = int()
    input_forget = int()

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i", "initial_c_i", "P_i"]
    output_params = ["Y_o", "Y_h_o", "Y_c_o"]
    attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "input_forget"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LSTM
        self.run_ = nn._LSTM_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.input_forget, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.initial_c_i, self.P_i, self.Y_o, self.Y_h_o, self.Y_c_o)

    def run(self):
        self.run_(self.name)

layer_map['LSTM'] = LSTM





class Identity:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Identity
        self.run_ = nn._Identity_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Identity'] = Identity





class Abs:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Abs
        self.run_ = nn._Abs_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Abs'] = Abs





class BatchNormalization:
    name = None
    X_i = str()
    scale_i = str()
    B_i = str()
    mean_i = str()
    var_i = str()
    Y_o = str()
    mean_o = str()
    var_o = str()
    saved_mean_o = str()
    saved_var_o = str()

    #parameters
    epsilon = float()
    momentum = float()

    input_params = ["X_i", "scale_i", "B_i", "mean_i", "var_i"]
    output_params = ["Y_o", "mean_o", "var_o", "saved_mean_o", "saved_var_o"]
    attribute_params = ["epsilon", "momentum"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._BatchNormalization
        self.run_ = nn._BatchNormalization_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.epsilon, self.momentum, self.X_i, self.scale_i, self.B_i, self.mean_i, self.var_i, self.Y_o, self.mean_o, self.var_o, self.saved_mean_o, self.saved_var_o)

    def run(self):
        self.run_(self.name)

layer_map['BatchNormalization'] = BatchNormalization





class Mean:
    name = None
    mean_o = str()

    #parameters

    input_params = []
    output_params = ["mean_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Mean
        self.run_ = nn._Mean_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.mean_o)

    def run(self):
        self.run_(self.name)

layer_map['Mean'] = Mean





class Add:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Add
        self.run_ = nn._Add_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Add'] = Add





class GlobalMaxPool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalMaxPool
        self.run_ = nn._GlobalMaxPool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['GlobalMaxPool'] = GlobalMaxPool





class Cast:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    to = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["to"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Cast
        self.run_ = nn._Cast_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.to, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Cast'] = Cast





class AveragePool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    kernel_shape = list()
    auto_pad = str()
    ceil_mode = int()
    count_include_pad = int()
    pads = list()
    strides = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["kernel_shape", "auto_pad", "ceil_mode", "count_include_pad", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._AveragePool
        self.run_ = nn._AveragePool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.ceil_mode, self.count_include_pad, self.pads, self.strides, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['AveragePool'] = AveragePool





class And:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._And
        self.run_ = nn._And_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['And'] = And





class LRN:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    size = int()
    alpha = float()
    beta = float()
    bias = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["size", "alpha", "beta", "bias"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LRN
        self.run_ = nn._LRN_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.size, self.alpha, self.beta, self.bias, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['LRN'] = LRN





class ArgMax:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axis = int()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axis", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ArgMax
        self.run_ = nn._ArgMax_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ArgMax'] = ArgMax





class Resize:
    name = None
    X_i = str()
    scales_i = str()
    Y_o = str()

    #parameters
    mode = str()

    input_params = ["X_i", "scales_i"]
    output_params = ["Y_o"]
    attribute_params = ["mode"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Resize
        self.run_ = nn._Resize_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.mode, self.X_i, self.scales_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Resize'] = Resize





class Expand:
    name = None
    input_i = str()
    shape_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i", "shape_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Expand
        self.run_ = nn._Expand_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.shape_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Expand'] = Expand





class Neg:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Neg
        self.run_ = nn._Neg_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Neg'] = Neg





class Mul:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Mul
        self.run_ = nn._Mul_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Mul'] = Mul





class ArgMin:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axis = int()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axis", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ArgMin
        self.run_ = nn._ArgMin_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ArgMin'] = ArgMin





class CastMap:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    cast_to = str()
    map_form = str()
    max_map = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["cast_to", "map_form", "max_map"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._CastMap
        self.run_ = nn._CastMap_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.cast_to, self.map_form, self.max_map, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['CastMap'] = CastMap





class Exp:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Exp
        self.run_ = nn._Exp_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Exp'] = Exp





class Div:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Div
        self.run_ = nn._Div_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Div'] = Div





class ReverseSequence:
    name = None
    input_i = str()
    sequence_lens_i = str()
    Y_o = str()

    #parameters
    batch_axis = int()
    time_axis = int()

    input_params = ["input_i", "sequence_lens_i"]
    output_params = ["Y_o"]
    attribute_params = ["batch_axis", "time_axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReverseSequence
        self.run_ = nn._ReverseSequence_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.batch_axis, self.time_axis, self.input_i, self.sequence_lens_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['ReverseSequence'] = ReverseSequence





class Ceil:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Ceil
        self.run_ = nn._Ceil_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Ceil'] = Ceil





class DepthToSpace:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    blocksize = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["blocksize"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._DepthToSpace
        self.run_ = nn._DepthToSpace_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.blocksize, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['DepthToSpace'] = DepthToSpace





class Clip:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    max = float()
    min = float()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["max", "min"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Clip
        self.run_ = nn._Clip_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.max, self.min, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Clip'] = Clip





class RNN:
    name = None
    X_i = str()
    W_i = str()
    R_i = str()
    B_i = str()
    sequence_lens_i = str()
    initial_h_i = str()
    Y_o = str()
    Y_h_o = str()

    #parameters
    activation_alpha = list()
    activation_beta = list()
    activations = list()
    clip = float()
    direction = str()
    hidden_size = int()

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i"]
    output_params = ["Y_o", "Y_h_o"]
    attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RNN
        self.run_ = nn._RNN_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.Y_o, self.Y_h_o)

    def run(self):
        self.run_(self.name)

layer_map['RNN'] = RNN





class Concat:
    name = None
    x0_i = str()
    x1_i = str()
    x2_i = str()
    x3_i = str()
    x4_i = str()
    x5_i = str()
    x6_i = str()
    x7_i = str()
    x8_i = str()
    x9_i = str()
    x10_i = str()
    x11_i = str()
    x12_i = str()
    x13_i = str()
    x14_i = str()
    x15_i = str()
    x16_i = str()
    x17_i = str()
    x18_i = str()
    x19_i = str()
    x20_i = str()
    x21_i = str()
    x22_i = str()
    x23_i = str()
    x24_i = str()
    x25_i = str()
    x26_i = str()
    x27_i = str()
    x28_i = str()
    x29_i = str()
    x30_i = str()
    x31_i = str()
    concat_result_o = str()

    #parameters
    axis = int()

    input_params = ["x0_i", "x1_i", "x2_i", "x3_i", "x4_i", "x5_i", "x6_i", "x7_i", "x8_i", "x9_i", "x10_i", "x11_i", "x12_i", "x13_i", "x14_i", "x15_i", "x16_i", "x17_i", "x18_i", "x19_i", "x20_i", "x21_i", "x22_i", "x23_i", "x24_i", "x25_i", "x26_i", "x27_i", "x28_i", "x29_i", "x30_i", "x31_i"]
    output_params = ["concat_result_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Concat
        self.run_ = nn._Concat_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.concat_result_o)

    def run(self):
        self.run_(self.name)

layer_map['Concat'] = Concat





class Constant:
    name = None
    output_o = str()

    #parameters
    value = list()

    input_params = []
    output_params = ["output_o"]
    attribute_params = ["value"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Constant
        self.run_ = nn._Constant_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.value, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Constant'] = Constant





class LpPool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    kernel_shape = list()
    auto_pad = str()
    p = int()
    pads = list()
    strides = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["kernel_shape", "auto_pad", "p", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LpPool
        self.run_ = nn._LpPool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.p, self.pads, self.strides, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['LpPool'] = LpPool





class Conv:
    name = None
    X_i = str()
    W_i = str()
    B_i = str()
    Y_o = str()

    #parameters
    auto_pad = str()
    dilations = list()
    group = int()
    kernel_shape = list()
    pads = list()
    strides = list()

    input_params = ["X_i", "W_i", "B_i"]
    output_params = ["Y_o"]
    attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Conv
        self.run_ = nn._Conv_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.X_i, self.W_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Conv'] = Conv





class Not:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Not
        self.run_ = nn._Not_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Not'] = Not





class Gather:
    name = None
    data_i = str()
    indices_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["data_i", "indices_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Gather
        self.run_ = nn._Gather_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.data_i, self.indices_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Gather'] = Gather





class ConvTranspose:
    name = None
    X_i = str()
    W_i = str()
    B_i = str()
    Y_o = str()

    #parameters
    auto_pad = str()
    dilations = list()
    group = int()
    kernel_shape = list()
    output_padding = list()
    output_shape = list()
    pads = list()
    strides = list()

    input_params = ["X_i", "W_i", "B_i"]
    output_params = ["Y_o"]
    attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "output_padding", "output_shape", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ConvTranspose
        self.run_ = nn._ConvTranspose_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.output_padding, self.output_shape, self.pads, self.strides, self.X_i, self.W_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['ConvTranspose'] = ConvTranspose





class Dropout:
    name = None
    data_i = str()
    output_o = str()
    mask_o = str()

    #parameters
    ratio = float()

    input_params = ["data_i"]
    output_params = ["output_o", "mask_o"]
    attribute_params = ["ratio"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Dropout
        self.run_ = nn._Dropout_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.ratio, self.data_i, self.output_o, self.mask_o)

    def run(self):
        self.run_(self.name)

layer_map['Dropout'] = Dropout





class LeakyRelu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    alpha = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LeakyRelu
        self.run_ = nn._LeakyRelu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['LeakyRelu'] = LeakyRelu





class Elu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    alpha = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Elu
        self.run_ = nn._Elu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Elu'] = Elu





class GlobalAveragePool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalAveragePool
        self.run_ = nn._GlobalAveragePool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['GlobalAveragePool'] = GlobalAveragePool





class Gemm:
    name = None
    A_i = str()
    B_i = str()
    C_i = str()
    Y_o = str()

    #parameters
    alpha = float()
    beta = float()
    transA = int()
    transB = int()

    input_params = ["A_i", "B_i", "C_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha", "beta", "transA", "transB"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Gemm
        self.run_ = nn._Gemm_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.beta, self.transA, self.transB, self.A_i, self.B_i, self.C_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Gemm'] = Gemm





class MaxPool:
    name = None
    X_i = str()
    Y_o = str()
    Indices_o = str()

    #parameters
    kernel_shape = list()
    auto_pad = str()
    ceil_mode = int()
    dilations = list()
    pads = list()
    storage_order = int()
    strides = list()

    input_params = ["X_i"]
    output_params = ["Y_o", "Indices_o"]
    attribute_params = ["kernel_shape", "auto_pad", "ceil_mode", "dilations", "pads", "storage_order", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxPool
        self.run_ = nn._MaxPool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.ceil_mode, self.dilations, self.pads, self.storage_order, self.strides, self.X_i, self.Y_o, self.Indices_o)

    def run(self):
        self.run_(self.name)

layer_map['MaxPool'] = MaxPool





class Equal:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Equal
        self.run_ = nn._Equal_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Equal'] = Equal





class Tile:
    name = None
    input_i = str()
    repeats_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i", "repeats_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Tile
        self.run_ = nn._Tile_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.repeats_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Tile'] = Tile





class Flatten:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Flatten
        self.run_ = nn._Flatten_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Flatten'] = Flatten





class Floor:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Floor
        self.run_ = nn._Floor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Floor'] = Floor





class GRU:
    name = None
    X_i = str()
    W_i = str()
    R_i = str()
    B_i = str()
    sequence_lens_i = str()
    initial_h_i = str()
    Y_o = str()
    Y_h_o = str()

    #parameters
    activation_alpha = list()
    activation_beta = list()
    activations = list()
    clip = float()
    direction = str()
    hidden_size = int()
    linear_before_reset = int()

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i"]
    output_params = ["Y_o", "Y_h_o"]
    attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "linear_before_reset"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._GRU
        self.run_ = nn._GRU_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.linear_before_reset, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.Y_o, self.Y_h_o)

    def run(self):
        self.run_(self.name)

layer_map['GRU'] = GRU





class GlobalLpPool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    p = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["p"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalLpPool
        self.run_ = nn._GlobalLpPool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.p, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['GlobalLpPool'] = GlobalLpPool





class Greater:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Greater
        self.run_ = nn._Greater_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Greater'] = Greater





class HardSigmoid:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    alpha = float()
    beta = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha", "beta"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._HardSigmoid
        self.run_ = nn._HardSigmoid_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.beta, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['HardSigmoid'] = HardSigmoid





class Selu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    alpha = float()
    gamma = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha", "gamma"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Selu
        self.run_ = nn._Selu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.gamma, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Selu'] = Selu





class Hardmax:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Hardmax
        self.run_ = nn._Hardmax_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Hardmax'] = Hardmax





class If:
    name = None
    cond_i = str()

    #parameters
    else_branch = int()
    then_branch = int()

    input_params = ["cond_i"]
    output_params = []
    attribute_params = ["else_branch", "then_branch"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._If
        self.run_ = nn._If_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.else_branch, self.then_branch, self.cond_i)

    def run(self):
        self.run_(self.name)

layer_map['If'] = If





class Min:
    name = None
    min_o = str()

    #parameters

    input_params = []
    output_params = ["min_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Min
        self.run_ = nn._Min_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.min_o)

    def run(self):
        self.run_(self.name)

layer_map['Min'] = Min





class InstanceNormalization:
    name = None
    input_i = str()
    scale_i = str()
    B_i = str()
    output_o = str()

    #parameters
    epsilon = float()

    input_params = ["input_i", "scale_i", "B_i"]
    output_params = ["output_o"]
    attribute_params = ["epsilon"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._InstanceNormalization
        self.run_ = nn._InstanceNormalization_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.epsilon, self.input_i, self.scale_i, self.B_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['InstanceNormalization'] = InstanceNormalization





class Less:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Less
        self.run_ = nn._Less_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Less'] = Less





class EyeLike:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    dtype = int()
    k = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["dtype", "k"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._EyeLike
        self.run_ = nn._EyeLike_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.dtype, self.k, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['EyeLike'] = EyeLike





class RandomNormal:
    name = None
    output_o = str()

    #parameters
    shape = list()
    dtype = int()
    mean = float()
    scale = float()
    seed = float()

    input_params = []
    output_params = ["output_o"]
    attribute_params = ["shape", "dtype", "mean", "scale", "seed"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomNormal
        self.run_ = nn._RandomNormal_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.shape, self.dtype, self.mean, self.scale, self.seed, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['RandomNormal'] = RandomNormal





class Slice:
    name = None
    data_i = str()
    starts_i = str()
    ends_i = str()
    axes_i = str()
    steps_i = str()
    output_o = str()

    #parameters

    input_params = ["data_i", "starts_i", "ends_i", "axes_i", "steps_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Slice
        self.run_ = nn._Slice_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.data_i, self.starts_i, self.ends_i, self.axes_i, self.steps_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Slice'] = Slice





class PRelu:
    name = None
    X_i = str()
    slope_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i", "slope_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._PRelu
        self.run_ = nn._PRelu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.slope_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['PRelu'] = PRelu





class Log:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Log
        self.run_ = nn._Log_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Log'] = Log





class LogSoftmax:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LogSoftmax
        self.run_ = nn._LogSoftmax_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['LogSoftmax'] = LogSoftmax





class Loop:
    name = None
    M_i = str()
    cond_i = str()

    #parameters
    body = int()

    input_params = ["M_i", "cond_i"]
    output_params = []
    attribute_params = ["body"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Loop
        self.run_ = nn._Loop_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.body, self.M_i, self.cond_i)

    def run(self):
        self.run_(self.name)

layer_map['Loop'] = Loop





class LpNormalization:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    axis = int()
    p = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["axis", "p"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LpNormalization
        self.run_ = nn._LpNormalization_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.p, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['LpNormalization'] = LpNormalization





class MatMul:
    name = None
    A_i = str()
    B_i = str()
    Y_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MatMul
        self.run_ = nn._MatMul_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['MatMul'] = MatMul





class ReduceL2:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceL2
        self.run_ = nn._ReduceL2_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceL2'] = ReduceL2





class Max:
    name = None
    max_o = str()

    #parameters

    input_params = []
    output_params = ["max_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Max
        self.run_ = nn._Max_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.max_o)

    def run(self):
        self.run_(self.name)

layer_map['Max'] = Max





class MaxRoiPool:
    name = None
    X_i = str()
    rois_i = str()
    Y_o = str()

    #parameters
    pooled_shape = list()
    spatial_scale = float()

    input_params = ["X_i", "rois_i"]
    output_params = ["Y_o"]
    attribute_params = ["pooled_shape", "spatial_scale"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxRoiPool
        self.run_ = nn._MaxRoiPool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.pooled_shape, self.spatial_scale, self.X_i, self.rois_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['MaxRoiPool'] = MaxRoiPool





class Or:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Or
        self.run_ = nn._Or_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Or'] = Or





class Pad:
    name = None
    data_i = str()
    output_o = str()

    #parameters
    pads = list()
    mode = str()
    value = float()

    input_params = ["data_i"]
    output_params = ["output_o"]
    attribute_params = ["pads", "mode", "value"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Pad
        self.run_ = nn._Pad_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.pads, self.mode, self.value, self.data_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Pad'] = Pad





class RandomUniformLike:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    dtype = int()
    high = float()
    low = float()
    seed = float()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["dtype", "high", "low", "seed"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomUniformLike
        self.run_ = nn._RandomUniformLike_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.dtype, self.high, self.low, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['RandomUniformLike'] = RandomUniformLike





class Reciprocal:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Reciprocal
        self.run_ = nn._Reciprocal_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Reciprocal'] = Reciprocal





class Pow:
    name = None
    X_i = str()
    Y_i = str()
    Z_o = str()

    #parameters

    input_params = ["X_i", "Y_i"]
    output_params = ["Z_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Pow
        self.run_ = nn._Pow_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_i, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['Pow'] = Pow





class RandomNormalLike:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    dtype = int()
    mean = float()
    scale = float()
    seed = float()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["dtype", "mean", "scale", "seed"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomNormalLike
        self.run_ = nn._RandomNormalLike_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.dtype, self.mean, self.scale, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['RandomNormalLike'] = RandomNormalLike





class OneHot:
    name = None
    indices_i = str()
    depth_i = str()
    values_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["indices_i", "depth_i", "values_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._OneHot
        self.run_ = nn._OneHot_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.indices_i, self.depth_i, self.values_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['OneHot'] = OneHot





class RandomUniform:
    name = None
    output_o = str()

    #parameters
    shape = list()
    dtype = int()
    high = float()
    low = float()
    seed = float()

    input_params = []
    output_params = ["output_o"]
    attribute_params = ["shape", "dtype", "high", "low", "seed"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomUniform
        self.run_ = nn._RandomUniform_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.shape, self.dtype, self.high, self.low, self.seed, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['RandomUniform'] = RandomUniform





class ReduceL1:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceL1
        self.run_ = nn._ReduceL1_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceL1'] = ReduceL1





class ReduceLogSum:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceLogSum
        self.run_ = nn._ReduceLogSum_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceLogSum'] = ReduceLogSum





class ReduceLogSumExp:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceLogSumExp
        self.run_ = nn._ReduceLogSumExp_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceLogSumExp'] = ReduceLogSumExp





class ReduceMax:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMax
        self.run_ = nn._ReduceMax_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceMax'] = ReduceMax





class OneHotEncoder:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    cats_int64s = list()
    cats_strings = list()
    zeros = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["cats_int64s", "cats_strings", "zeros"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._OneHotEncoder
        self.run_ = nn._OneHotEncoder_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.cats_int64s, self.cats_strings, self.zeros, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['OneHotEncoder'] = OneHotEncoder





class IsNaN:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._IsNaN
        self.run_ = nn._IsNaN_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['IsNaN'] = IsNaN





class ReduceMean:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMean
        self.run_ = nn._ReduceMean_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceMean'] = ReduceMean





class ReduceMin:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMin
        self.run_ = nn._ReduceMin_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceMin'] = ReduceMin





class TreeEnsembleRegressor:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    aggregate_function = str()
    base_values = list()
    n_targets = int()
    nodes_falsenodeids = list()
    nodes_featureids = list()
    nodes_hitrates = list()
    nodes_missing_value_tracks_true = list()
    nodes_modes = list()
    nodes_nodeids = list()
    nodes_treeids = list()
    nodes_truenodeids = list()
    nodes_values = list()
    post_transform = str()
    target_ids = list()
    target_nodeids = list()
    target_treeids = list()
    target_weights = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["aggregate_function", "base_values", "n_targets", "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates", "nodes_missing_value_tracks_true", "nodes_modes", "nodes_nodeids", "nodes_treeids", "nodes_truenodeids", "nodes_values", "post_transform", "target_ids", "target_nodeids", "target_treeids", "target_weights"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._TreeEnsembleRegressor
        self.run_ = nn._TreeEnsembleRegressor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.aggregate_function, self.base_values, self.n_targets, self.nodes_falsenodeids, self.nodes_featureids, self.nodes_hitrates, self.nodes_missing_value_tracks_true, self.nodes_modes, self.nodes_nodeids, self.nodes_treeids, self.nodes_truenodeids, self.nodes_values, self.post_transform, self.target_ids, self.target_nodeids, self.target_treeids, self.target_weights, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['TreeEnsembleRegressor'] = TreeEnsembleRegressor





class ReduceProd:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceProd
        self.run_ = nn._ReduceProd_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceProd'] = ReduceProd





class ReduceSum:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceSum
        self.run_ = nn._ReduceSum_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceSum'] = ReduceSum





class ReduceSumSquare:
    name = None
    data_i = str()
    reduced_o = str()

    #parameters
    axes = list()
    keepdims = int()

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    attribute_params = ["axes", "keepdims"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceSumSquare
        self.run_ = nn._ReduceSumSquare_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

layer_map['ReduceSumSquare'] = ReduceSumSquare





class Relu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Relu
        self.run_ = nn._Relu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Relu'] = Relu





class Reshape:
    name = None
    data_i = str()
    shape_i = str()
    reshaped_o = str()

    #parameters

    input_params = ["data_i", "shape_i"]
    output_params = ["reshaped_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Reshape
        self.run_ = nn._Reshape_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.data_i, self.shape_i, self.reshaped_o)

    def run(self):
        self.run_(self.name)

layer_map['Reshape'] = Reshape





class Shape:
    name = None
    data_i = str()
    shape_o = str()

    #parameters

    input_params = ["data_i"]
    output_params = ["shape_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Shape
        self.run_ = nn._Shape_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.data_i, self.shape_o)

    def run(self):
        self.run_(self.name)

layer_map['Shape'] = Shape





class Sigmoid:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sigmoid
        self.run_ = nn._Sigmoid_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Sigmoid'] = Sigmoid





class Size:
    name = None
    data_i = str()
    size_o = str()

    #parameters

    input_params = ["data_i"]
    output_params = ["size_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Size
        self.run_ = nn._Size_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.data_i, self.size_o)

    def run(self):
        self.run_(self.name)

layer_map['Size'] = Size





class Softmax:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Softmax
        self.run_ = nn._Softmax_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Softmax'] = Softmax





class Softplus:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Softplus
        self.run_ = nn._Softplus_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Softplus'] = Softplus





class Softsign:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Softsign
        self.run_ = nn._Softsign_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Softsign'] = Softsign





class SpaceToDepth:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    blocksize = int()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["blocksize"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._SpaceToDepth
        self.run_ = nn._SpaceToDepth_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.blocksize, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['SpaceToDepth'] = SpaceToDepth





class TfIdfVectorizer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    max_gram_length = int()
    max_skip_count = int()
    min_gram_length = int()
    mode = str()
    ngram_counts = list()
    ngram_indexes = list()
    pool_int64s = list()
    pool_strings = list()
    weights = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["max_gram_length", "max_skip_count", "min_gram_length", "mode", "ngram_counts", "ngram_indexes", "pool_int64s", "pool_strings", "weights"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._TfIdfVectorizer
        self.run_ = nn._TfIdfVectorizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.max_gram_length, self.max_skip_count, self.min_gram_length, self.mode, self.ngram_counts, self.ngram_indexes, self.pool_int64s, self.pool_strings, self.weights, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['TfIdfVectorizer'] = TfIdfVectorizer





class Split:
    name = None
    input_i = str()

    #parameters
    axis = int()
    split = list()

    input_params = ["input_i"]
    output_params = []
    attribute_params = ["axis", "split"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Split
        self.run_ = nn._Split_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.split, self.input_i)

    def run(self):
        self.run_(self.name)

layer_map['Split'] = Split





class Imputer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    imputed_value_floats = list()
    imputed_value_int64s = list()
    replaced_value_float = float()
    replaced_value_int64 = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["imputed_value_floats", "imputed_value_int64s", "replaced_value_float", "replaced_value_int64"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Imputer
        self.run_ = nn._Imputer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.imputed_value_floats, self.imputed_value_int64s, self.replaced_value_float, self.replaced_value_int64, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Imputer'] = Imputer





class Sqrt:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sqrt
        self.run_ = nn._Sqrt_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Sqrt'] = Sqrt





class Squeeze:
    name = None
    data_i = str()
    squeezed_o = str()

    #parameters
    axes = list()

    input_params = ["data_i"]
    output_params = ["squeezed_o"]
    attribute_params = ["axes"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Squeeze
        self.run_ = nn._Squeeze_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.data_i, self.squeezed_o)

    def run(self):
        self.run_(self.name)

layer_map['Squeeze'] = Squeeze





class TopK:
    name = None
    X_i = str()
    K_i = str()
    Values_o = str()
    Indices_o = str()

    #parameters
    axis = int()

    input_params = ["X_i", "K_i"]
    output_params = ["Values_o", "Indices_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._TopK
        self.run_ = nn._TopK_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.X_i, self.K_i, self.Values_o, self.Indices_o)

    def run(self):
        self.run_(self.name)

layer_map['TopK'] = TopK





class Sub:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sub
        self.run_ = nn._Sub_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Sub'] = Sub





class Sum:
    name = None
    x0_i = str()
    x1_i = str()
    x2_i = str()
    x3_i = str()
    x4_i = str()
    x5_i = str()
    x6_i = str()
    x7_i = str()
    x8_i = str()
    x9_i = str()
    x10_i = str()
    x11_i = str()
    x12_i = str()
    x13_i = str()
    x14_i = str()
    x15_i = str()
    x16_i = str()
    x17_i = str()
    x18_i = str()
    x19_i = str()
    x20_i = str()
    x21_i = str()
    x22_i = str()
    x23_i = str()
    x24_i = str()
    x25_i = str()
    x26_i = str()
    x27_i = str()
    x28_i = str()
    x29_i = str()
    x30_i = str()
    x31_i = str()
    sum_o = str()

    #parameters

    input_params = ["x0_i", "x1_i", "x2_i", "x3_i", "x4_i", "x5_i", "x6_i", "x7_i", "x8_i", "x9_i", "x10_i", "x11_i", "x12_i", "x13_i", "x14_i", "x15_i", "x16_i", "x17_i", "x18_i", "x19_i", "x20_i", "x21_i", "x22_i", "x23_i", "x24_i", "x25_i", "x26_i", "x27_i", "x28_i", "x29_i", "x30_i", "x31_i"]
    output_params = ["sum_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sum
        self.run_ = nn._Sum_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.sum_o)

    def run(self):
        self.run_(self.name)

layer_map['Sum'] = Sum





class Shrink:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    bias = float()
    lambd = float()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["bias", "lambd"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Shrink
        self.run_ = nn._Shrink_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.bias, self.lambd, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Shrink'] = Shrink





class Tanh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Tanh
        self.run_ = nn._Tanh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Tanh'] = Tanh





class Transpose:
    name = None
    data_i = str()
    transposed_o = str()

    #parameters
    perm = list()

    input_params = ["data_i"]
    output_params = ["transposed_o"]
    attribute_params = ["perm"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Transpose
        self.run_ = nn._Transpose_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.perm, self.data_i, self.transposed_o)

    def run(self):
        self.run_(self.name)

layer_map['Transpose'] = Transpose





class Unsqueeze:
    name = None
    data_i = str()
    expanded_o = str()

    #parameters
    axes = list()

    input_params = ["data_i"]
    output_params = ["expanded_o"]
    attribute_params = ["axes"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Unsqueeze
        self.run_ = nn._Unsqueeze_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.data_i, self.expanded_o)

    def run(self):
        self.run_(self.name)

layer_map['Unsqueeze'] = Unsqueeze





class Upsample:
    name = None
    X_i = str()
    scales_i = str()
    Y_o = str()

    #parameters
    mode = str()

    input_params = ["X_i", "scales_i"]
    output_params = ["Y_o"]
    attribute_params = ["mode"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Upsample
        self.run_ = nn._Upsample_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.mode, self.X_i, self.scales_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Upsample'] = Upsample





class SVMClassifier:
    name = None
    X_i = str()
    Y_o = str()
    Z_o = str()

    #parameters
    classlabels_ints = list()
    classlabels_strings = list()
    coefficients = list()
    kernel_params = list()
    kernel_type = str()
    post_transform = str()
    prob_a = list()
    prob_b = list()
    rho = list()
    support_vectors = list()
    vectors_per_class = list()

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    attribute_params = ["classlabels_ints", "classlabels_strings", "coefficients", "kernel_params", "kernel_type", "post_transform", "prob_a", "prob_b", "rho", "support_vectors", "vectors_per_class"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._SVMClassifier
        self.run_ = nn._SVMClassifier_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.classlabels_ints, self.classlabels_strings, self.coefficients, self.kernel_params, self.kernel_type, self.post_transform, self.prob_a, self.prob_b, self.rho, self.support_vectors, self.vectors_per_class, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['SVMClassifier'] = SVMClassifier





class Xor:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Xor
        self.run_ = nn._Xor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Xor'] = Xor





class Acos:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Acos
        self.run_ = nn._Acos_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Acos'] = Acos





class Asin:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Asin
        self.run_ = nn._Asin_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Asin'] = Asin





class Atan:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Atan
        self.run_ = nn._Atan_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Atan'] = Atan





class Cos:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Cos
        self.run_ = nn._Cos_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Cos'] = Cos





class Sin:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sin
        self.run_ = nn._Sin_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Sin'] = Sin





class Tan:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Tan
        self.run_ = nn._Tan_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Tan'] = Tan





class Multinomial:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    dtype = int()
    sample_size = int()
    seed = float()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["dtype", "sample_size", "seed"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Multinomial
        self.run_ = nn._Multinomial_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.dtype, self.sample_size, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Multinomial'] = Multinomial





class Scan:
    name = None
    x0_i = str()
    x1_i = str()
    x2_i = str()
    x3_i = str()
    x4_i = str()
    x5_i = str()
    x6_i = str()
    x7_i = str()
    x8_i = str()
    x9_i = str()
    x10_i = str()
    x11_i = str()
    x12_i = str()
    x13_i = str()
    x14_i = str()
    x15_i = str()
    x16_i = str()
    x17_i = str()
    x18_i = str()
    x19_i = str()
    x20_i = str()
    x21_i = str()
    x22_i = str()
    x23_i = str()
    x24_i = str()
    x25_i = str()
    x26_i = str()
    x27_i = str()
    x28_i = str()
    x29_i = str()
    x30_i = str()
    x31_i = str()
    y0_o = str()
    y1_o = str()
    y2_o = str()
    y3_o = str()
    y4_o = str()
    y5_o = str()
    y6_o = str()
    y7_o = str()
    y8_o = str()
    y9_o = str()
    y10_o = str()
    y11_o = str()
    y12_o = str()
    y13_o = str()
    y14_o = str()
    y15_o = str()
    y16_o = str()
    y17_o = str()
    y18_o = str()
    y19_o = str()
    y20_o = str()
    y21_o = str()
    y22_o = str()
    y23_o = str()
    y24_o = str()
    y25_o = str()
    y26_o = str()
    y27_o = str()
    y28_o = str()
    y29_o = str()
    y30_o = str()
    y31_o = str()

    #parameters
    body = int()
    num_scan_inputs = int()
    scan_input_axes = list()
    scan_input_directions = list()
    scan_output_axes = list()
    scan_output_directions = list()

    input_params = ["x0_i", "x1_i", "x2_i", "x3_i", "x4_i", "x5_i", "x6_i", "x7_i", "x8_i", "x9_i", "x10_i", "x11_i", "x12_i", "x13_i", "x14_i", "x15_i", "x16_i", "x17_i", "x18_i", "x19_i", "x20_i", "x21_i", "x22_i", "x23_i", "x24_i", "x25_i", "x26_i", "x27_i", "x28_i", "x29_i", "x30_i", "x31_i"]
    output_params = ["y0_o", "y1_o", "y2_o", "y3_o", "y4_o", "y5_o", "y6_o", "y7_o", "y8_o", "y9_o", "y10_o", "y11_o", "y12_o", "y13_o", "y14_o", "y15_o", "y16_o", "y17_o", "y18_o", "y19_o", "y20_o", "y21_o", "y22_o", "y23_o", "y24_o", "y25_o", "y26_o", "y27_o", "y28_o", "y29_o", "y30_o", "y31_o"]
    attribute_params = ["body", "num_scan_inputs", "scan_input_axes", "scan_input_directions", "scan_output_axes", "scan_output_directions"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Scan
        self.run_ = nn._Scan_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.body, self.num_scan_inputs, self.scan_input_axes, self.scan_input_directions, self.scan_output_axes, self.scan_output_directions, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.y0_o, self.y1_o, self.y2_o, self.y3_o, self.y4_o, self.y5_o, self.y6_o, self.y7_o, self.y8_o, self.y9_o, self.y10_o, self.y11_o, self.y12_o, self.y13_o, self.y14_o, self.y15_o, self.y16_o, self.y17_o, self.y18_o, self.y19_o, self.y20_o, self.y21_o, self.y22_o, self.y23_o, self.y24_o, self.y25_o, self.y26_o, self.y27_o, self.y28_o, self.y29_o, self.y30_o, self.y31_o)

    def run(self):
        self.run_(self.name)

layer_map['Scan'] = Scan





class Compress:
    name = None
    input_i = str()
    condition_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["input_i", "condition_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Compress
        self.run_ = nn._Compress_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.condition_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Compress'] = Compress





class ConstantOfShape:
    name = None
    input_i = str()
    output_o = str()

    #parameters
    value = list()

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = ["value"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ConstantOfShape
        self.run_ = nn._ConstantOfShape_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.value, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['ConstantOfShape'] = ConstantOfShape





class MaxUnpool:
    name = None
    X_i = str()
    I_i = str()
    output_shape_i = str()
    output_o = str()

    #parameters
    kernel_shape = list()
    pads = list()
    strides = list()

    input_params = ["X_i", "I_i", "output_shape_i"]
    output_params = ["output_o"]
    attribute_params = ["kernel_shape", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxUnpool
        self.run_ = nn._MaxUnpool_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.kernel_shape, self.pads, self.strides, self.X_i, self.I_i, self.output_shape_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['MaxUnpool'] = MaxUnpool





class Scatter:
    name = None
    data_i = str()
    indices_i = str()
    updates_i = str()
    output_o = str()

    #parameters
    axis = int()

    input_params = ["data_i", "indices_i", "updates_i"]
    output_params = ["output_o"]
    attribute_params = ["axis"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Scatter
        self.run_ = nn._Scatter_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axis, self.data_i, self.indices_i, self.updates_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Scatter'] = Scatter





class Sinh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sinh
        self.run_ = nn._Sinh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Sinh'] = Sinh





class Cosh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Cosh
        self.run_ = nn._Cosh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Cosh'] = Cosh





class Asinh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Asinh
        self.run_ = nn._Asinh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Asinh'] = Asinh





class Acosh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Acosh
        self.run_ = nn._Acosh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Acosh'] = Acosh





class NonMaxSuppression:
    name = None
    boxes_i = str()
    scores_i = str()
    max_output_boxes_per_class_i = str()
    iou_threshold_i = str()
    score_threshold_i = str()
    selected_indices_o = str()

    #parameters
    center_point_box = int()

    input_params = ["boxes_i", "scores_i", "max_output_boxes_per_class_i", "iou_threshold_i", "score_threshold_i"]
    output_params = ["selected_indices_o"]
    attribute_params = ["center_point_box"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._NonMaxSuppression
        self.run_ = nn._NonMaxSuppression_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.center_point_box, self.boxes_i, self.scores_i, self.max_output_boxes_per_class_i, self.iou_threshold_i, self.score_threshold_i, self.selected_indices_o)

    def run(self):
        self.run_(self.name)

layer_map['NonMaxSuppression'] = NonMaxSuppression





class Atanh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Atanh
        self.run_ = nn._Atanh_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Atanh'] = Atanh





class Sign:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Sign
        self.run_ = nn._Sign_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Sign'] = Sign





class Erf:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Erf
        self.run_ = nn._Erf_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Erf'] = Erf





class Where:
    name = None
    condition_i = str()
    X_i = str()
    Y_i = str()
    output_o = str()

    #parameters

    input_params = ["condition_i", "X_i", "Y_i"]
    output_params = ["output_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Where
        self.run_ = nn._Where_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.condition_i, self.X_i, self.Y_i, self.output_o)

    def run(self):
        self.run_(self.name)

layer_map['Where'] = Where





class NonZero:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._NonZero
        self.run_ = nn._NonZero_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['NonZero'] = NonZero





class MeanVarianceNormalization:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    axes = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["axes"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MeanVarianceNormalization
        self.run_ = nn._MeanVarianceNormalization_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.axes, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['MeanVarianceNormalization'] = MeanVarianceNormalization





class StringNormalizer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    case_change_action = str()
    is_case_sensitive = int()
    locale = str()
    stopwords = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["case_change_action", "is_case_sensitive", "locale", "stopwords"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._StringNormalizer
        self.run_ = nn._StringNormalizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.case_change_action, self.is_case_sensitive, self.locale, self.stopwords, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['StringNormalizer'] = StringNormalizer





class Mod:
    name = None
    A_i = str()
    B_i = str()
    C_o = str()

    #parameters
    fmod = int()

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    attribute_params = ["fmod"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Mod
        self.run_ = nn._Mod_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.fmod, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

layer_map['Mod'] = Mod





class ThresholdedRelu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    alpha = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["alpha"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ThresholdedRelu
        self.run_ = nn._ThresholdedRelu_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['ThresholdedRelu'] = ThresholdedRelu





class MatMulInteger:
    name = None
    A_i = str()
    B_i = str()
    a_zero_point_i = str()
    b_zero_point_i = str()
    Y_o = str()

    #parameters

    input_params = ["A_i", "B_i", "a_zero_point_i", "b_zero_point_i"]
    output_params = ["Y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._MatMulInteger
        self.run_ = nn._MatMulInteger_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.a_zero_point_i, self.b_zero_point_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['MatMulInteger'] = MatMulInteger





class QLinearMatMul:
    name = None
    a_i = str()
    a_scale_i = str()
    a_zero_point_i = str()
    b_i = str()
    b_scale_i = str()
    b_zero_point_i = str()
    y_scale_i = str()
    y_zero_point_i = str()
    y_o = str()

    #parameters

    input_params = ["a_i", "a_scale_i", "a_zero_point_i", "b_i", "b_scale_i", "b_zero_point_i", "y_scale_i", "y_zero_point_i"]
    output_params = ["y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._QLinearMatMul
        self.run_ = nn._QLinearMatMul_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.a_i, self.a_scale_i, self.a_zero_point_i, self.b_i, self.b_scale_i, self.b_zero_point_i, self.y_scale_i, self.y_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

layer_map['QLinearMatMul'] = QLinearMatMul





class ConvInteger:
    name = None
    x_i = str()
    w_i = str()
    x_zero_point_i = str()
    w_zero_point_i = str()
    y_o = str()

    #parameters
    auto_pad = str()
    dilations = list()
    group = int()
    kernel_shape = list()
    pads = list()
    strides = list()

    input_params = ["x_i", "w_i", "x_zero_point_i", "w_zero_point_i"]
    output_params = ["y_o"]
    attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ConvInteger
        self.run_ = nn._ConvInteger_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.x_i, self.w_i, self.x_zero_point_i, self.w_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

layer_map['ConvInteger'] = ConvInteger





class QLinearConv:
    name = None
    x_i = str()
    x_scale_i = str()
    x_zero_point_i = str()
    w_i = str()
    w_scale_i = str()
    w_zero_point_i = str()
    y_scale_i = str()
    y_zero_point_i = str()
    B_i = str()
    y_o = str()

    #parameters
    auto_pad = str()
    dilations = list()
    group = int()
    kernel_shape = list()
    pads = list()
    strides = list()

    input_params = ["x_i", "x_scale_i", "x_zero_point_i", "w_i", "w_scale_i", "w_zero_point_i", "y_scale_i", "y_zero_point_i", "B_i"]
    output_params = ["y_o"]
    attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._QLinearConv
        self.run_ = nn._QLinearConv_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.x_i, self.x_scale_i, self.x_zero_point_i, self.w_i, self.w_scale_i, self.w_zero_point_i, self.y_scale_i, self.y_zero_point_i, self.B_i, self.y_o)

    def run(self):
        self.run_(self.name)

layer_map['QLinearConv'] = QLinearConv





class QuantizeLinear:
    name = None
    x_i = str()
    y_scale_i = str()
    y_zero_point_i = str()
    y_o = str()

    #parameters

    input_params = ["x_i", "y_scale_i", "y_zero_point_i"]
    output_params = ["y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._QuantizeLinear
        self.run_ = nn._QuantizeLinear_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.x_i, self.y_scale_i, self.y_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

layer_map['QuantizeLinear'] = QuantizeLinear





class DequantizeLinear:
    name = None
    x_i = str()
    x_scale_i = str()
    x_zero_point_i = str()
    y_o = str()

    #parameters

    input_params = ["x_i", "x_scale_i", "x_zero_point_i"]
    output_params = ["y_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._DequantizeLinear
        self.run_ = nn._DequantizeLinear_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.x_i, self.x_scale_i, self.x_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

layer_map['DequantizeLinear'] = DequantizeLinear





class IsInf:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    detect_negative = int()
    detect_positive = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["detect_negative", "detect_positive"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._IsInf
        self.run_ = nn._IsInf_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.detect_negative, self.detect_positive, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['IsInf'] = IsInf





class RoiAlign:
    name = None
    X_i = str()
    rois_i = str()
    batch_indices_i = str()
    Y_o = str()

    #parameters
    mode = str()
    output_height = int()
    output_width = int()
    sampling_ratio = int()
    spatial_scale = float()

    input_params = ["X_i", "rois_i", "batch_indices_i"]
    output_params = ["Y_o"]
    attribute_params = ["mode", "output_height", "output_width", "sampling_ratio", "spatial_scale"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._RoiAlign
        self.run_ = nn._RoiAlign_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.mode, self.output_height, self.output_width, self.sampling_ratio, self.spatial_scale, self.X_i, self.rois_i, self.batch_indices_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['RoiAlign'] = RoiAlign





class ArrayFeatureExtractor:
    name = None
    X_i = str()
    Y_i = str()
    Z_o = str()

    #parameters

    input_params = ["X_i", "Y_i"]
    output_params = ["Z_o"]
    attribute_params = []
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ArrayFeatureExtractor
        self.run_ = nn._ArrayFeatureExtractor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.X_i, self.Y_i, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['ArrayFeatureExtractor'] = ArrayFeatureExtractor





class Binarizer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    threshold = float()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["threshold"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Binarizer
        self.run_ = nn._Binarizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.threshold, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Binarizer'] = Binarizer





class CategoryMapper:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    cats_int64s = list()
    cats_strings = list()
    default_int64 = int()
    default_string = str()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["cats_int64s", "cats_strings", "default_int64", "default_string"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._CategoryMapper
        self.run_ = nn._CategoryMapper_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.cats_int64s, self.cats_strings, self.default_int64, self.default_string, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['CategoryMapper'] = CategoryMapper





class DictVectorizer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    int64_vocabulary = list()
    string_vocabulary = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["int64_vocabulary", "string_vocabulary"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._DictVectorizer
        self.run_ = nn._DictVectorizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.int64_vocabulary, self.string_vocabulary, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['DictVectorizer'] = DictVectorizer





class FeatureVectorizer:
    name = None
    Y_o = str()

    #parameters
    inputdimensions = list()

    input_params = []
    output_params = ["Y_o"]
    attribute_params = ["inputdimensions"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._FeatureVectorizer
        self.run_ = nn._FeatureVectorizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.inputdimensions, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['FeatureVectorizer'] = FeatureVectorizer





class LabelEncoder:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    default_float = float()
    default_int64 = int()
    default_string = str()
    keys_floats = list()
    keys_int64s = list()
    keys_strings = list()
    values_floats = list()
    values_int64s = list()
    values_strings = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["default_float", "default_int64", "default_string", "keys_floats", "keys_int64s", "keys_strings", "values_floats", "values_int64s", "values_strings"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LabelEncoder
        self.run_ = nn._LabelEncoder_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.default_float, self.default_int64, self.default_string, self.keys_floats, self.keys_int64s, self.keys_strings, self.values_floats, self.values_int64s, self.values_strings, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['LabelEncoder'] = LabelEncoder





class LinearClassifier:
    name = None
    X_i = str()
    Y_o = str()
    Z_o = str()

    #parameters
    coefficients = list()
    classlabels_ints = list()
    classlabels_strings = list()
    intercepts = list()
    multi_class = int()
    post_transform = str()

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    attribute_params = ["coefficients", "classlabels_ints", "classlabels_strings", "intercepts", "multi_class", "post_transform"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LinearClassifier
        self.run_ = nn._LinearClassifier_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.coefficients, self.classlabels_ints, self.classlabels_strings, self.intercepts, self.multi_class, self.post_transform, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['LinearClassifier'] = LinearClassifier





class LinearRegressor:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    coefficients = list()
    intercepts = list()
    post_transform = str()
    targets = int()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["coefficients", "intercepts", "post_transform", "targets"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._LinearRegressor
        self.run_ = nn._LinearRegressor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.coefficients, self.intercepts, self.post_transform, self.targets, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['LinearRegressor'] = LinearRegressor





class Normalizer:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    norm = str()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["norm"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Normalizer
        self.run_ = nn._Normalizer_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.norm, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Normalizer'] = Normalizer





class SVMRegressor:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    coefficients = list()
    kernel_params = list()
    kernel_type = str()
    n_supports = int()
    one_class = int()
    post_transform = str()
    rho = list()
    support_vectors = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["coefficients", "kernel_params", "kernel_type", "n_supports", "one_class", "post_transform", "rho", "support_vectors"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._SVMRegressor
        self.run_ = nn._SVMRegressor_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.coefficients, self.kernel_params, self.kernel_type, self.n_supports, self.one_class, self.post_transform, self.rho, self.support_vectors, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['SVMRegressor'] = SVMRegressor





class Scaler:
    name = None
    X_i = str()
    Y_o = str()

    #parameters
    offset = list()
    scale = list()

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = ["offset", "scale"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._Scaler
        self.run_ = nn._Scaler_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.offset, self.scale, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

layer_map['Scaler'] = Scaler





class TreeEnsembleClassifier:
    name = None
    X_i = str()
    Y_o = str()
    Z_o = str()

    #parameters
    base_values = list()
    class_ids = list()
    class_nodeids = list()
    class_treeids = list()
    class_weights = list()
    classlabels_int64s = list()
    classlabels_strings = list()
    nodes_falsenodeids = list()
    nodes_featureids = list()
    nodes_hitrates = list()
    nodes_missing_value_tracks_true = list()
    nodes_modes = list()
    nodes_nodeids = list()
    nodes_treeids = list()
    nodes_truenodeids = list()
    nodes_values = list()
    post_transform = str()

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    attribute_params = ["base_values", "class_ids", "class_nodeids", "class_treeids", "class_weights", "classlabels_int64s", "classlabels_strings", "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates", "nodes_missing_value_tracks_true", "nodes_modes", "nodes_nodeids", "nodes_treeids", "nodes_truenodeids", "nodes_values", "post_transform"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._TreeEnsembleClassifier
        self.run_ = nn._TreeEnsembleClassifier_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.base_values, self.class_ids, self.class_nodeids, self.class_treeids, self.class_weights, self.classlabels_int64s, self.classlabels_strings, self.nodes_falsenodeids, self.nodes_featureids, self.nodes_hitrates, self.nodes_missing_value_tracks_true, self.nodes_modes, self.nodes_nodeids, self.nodes_treeids, self.nodes_truenodeids, self.nodes_values, self.post_transform, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['TreeEnsembleClassifier'] = TreeEnsembleClassifier





class ZipMap:
    name = None
    X_i = str()
    Z_o = str()

    #parameters
    classlabels_int64s = list()
    classlabels_strings = list()

    input_params = ["X_i"]
    output_params = ["Z_o"]
    attribute_params = ["classlabels_int64s", "classlabels_strings"]
    i   = ''
    o   = ''
    def __init__(self, name):
        self.name = name
        self.Module = nn._ZipMap
        self.run_ = nn._ZipMap_run

        

    def output_shape(self, tensor):
        return tensor[self.__dict__[self.input_params[0]]].shape 

    def input(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x
            if(self.i != ''):
                self.i = x
            
    def output(self, tensors, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape(tensors))
                if(self.o != ''):
                    self.o = x


    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def build(self):
        self.Module(self.name, self.classlabels_int64s, self.classlabels_strings, self.X_i, self.Z_o)

    def run(self):
        self.run_(self.name)

layer_map['ZipMap'] = ZipMap

