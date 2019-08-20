#import _backend.nn as nn
layer_map = {}


class LSTM:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_i = None
    W_i = None
    R_i = None
    B_i = None
    sequence_lens_i = None
    initial_h_i = None
    initial_c_i = None
    P_i = None
    Y_o = None
    Y_h_o = None
    Y_c_o = None

    #parameters
    clip = None
    direction = None
    hidden_size = None
    input_forget = None

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i", "initial_c_i", "P_i"]
    output_params = ["Y_o", "Y_h_o", "Y_c_o"]
    #attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "input_forget"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LSTM(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LSTM'] = LSTM





class Identity:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Identity(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Identity'] = Identity





class Abs:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Abs(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Abs'] = Abs





class BatchNormalization:
    name = None
    X_i = None
    scale_i = None
    B_i = None
    mean_i = None
    var_i = None
    Y_o = None
    mean_o = None
    var_o = None
    saved_mean_o = None
    saved_var_o = None

    #parameters
    epsilon = None
    momentum = None

    input_params = ["X_i", "scale_i", "B_i", "mean_i", "var_i"]
    output_params = ["Y_o", "mean_o", "var_o", "saved_mean_o", "saved_var_o"]
    #attribute_params = ["epsilon", "momentum"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._BatchNormalization(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['BatchNormalization'] = BatchNormalization





class Mean:
    name = None
    mean_o = None

    #parameters

    input_params = []
    output_params = ["mean_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Mean(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Mean'] = Mean





class Add:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Add(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Add'] = Add





class GlobalMaxPool:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._GlobalMaxPool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['GlobalMaxPool'] = GlobalMaxPool





class Cast:
    name = None
    input_i = None
    output_o = None

    #parameters
    to = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["to"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Cast(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Cast'] = Cast





class AveragePool:
    name = None
    X_i = None
    Y_o = None

    #parameters
    kernel_shape = None
    auto_pad = None
    ceil_mode = None
    count_include_pad = None
    pads = None
    strides = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["kernel_shape", "auto_pad", "ceil_mode", "count_include_pad", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._AveragePool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['AveragePool'] = AveragePool





class And:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._And(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['And'] = And





class LRN:
    name = None
    X_i = None
    Y_o = None

    #parameters
    size = None
    alpha = None
    beta = None
    bias = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["size", "alpha", "beta", "bias"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LRN(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LRN'] = LRN





class ArgMax:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axis = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axis", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ArgMax(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ArgMax'] = ArgMax





class Resize:
    name = None
    X_i = None
    scales_i = None
    Y_o = None

    #parameters
    mode = None

    input_params = ["X_i", "scales_i"]
    output_params = ["Y_o"]
    #attribute_params = ["mode"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Resize(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Resize'] = Resize





class Expand:
    name = None
    input_i = None
    shape_i = None
    output_o = None

    #parameters

    input_params = ["input_i", "shape_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Expand(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Expand'] = Expand





class Neg:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Neg(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Neg'] = Neg





class Mul:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Mul(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Mul'] = Mul





class ArgMin:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axis = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axis", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ArgMin(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ArgMin'] = ArgMin





class CastMap:
    name = None
    X_i = None
    Y_o = None

    #parameters
    cast_to = None
    map_form = None
    max_map = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["cast_to", "map_form", "max_map"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._CastMap(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['CastMap'] = CastMap





class Exp:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Exp(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Exp'] = Exp





class Div:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Div(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Div'] = Div





class ReverseSequence:
    name = None
    input_i = None
    sequence_lens_i = None
    Y_o = None

    #parameters
    batch_axis = None
    time_axis = None

    input_params = ["input_i", "sequence_lens_i"]
    output_params = ["Y_o"]
    #attribute_params = ["batch_axis", "time_axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReverseSequence(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReverseSequence'] = ReverseSequence





class Ceil:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Ceil(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Ceil'] = Ceil





class DepthToSpace:
    name = None
    input_i = None
    output_o = None

    #parameters
    blocksize = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["blocksize"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._DepthToSpace(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['DepthToSpace'] = DepthToSpace





class Clip:
    name = None
    input_i = None
    output_o = None

    #parameters
    max = None
    min = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["max", "min"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Clip(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Clip'] = Clip





class RNN:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_i = None
    W_i = None
    R_i = None
    B_i = None
    sequence_lens_i = None
    initial_h_i = None
    Y_o = None
    Y_h_o = None

    #parameters
    clip = None
    direction = None
    hidden_size = None

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i"]
    output_params = ["Y_o", "Y_h_o"]
    #attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RNN(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RNN'] = RNN





class Concat:
    name = None
    concat_result_o = None

    #parameters
    axis = None

    input_params = []
    output_params = ["concat_result_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Concat(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Concat'] = Concat





class Constant:
    name = None
    value = None
    output_o = None

    #parameters

    input_params = []
    output_params = ["output_o"]
    #attribute_params = ["value"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Constant(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Constant'] = Constant





class LpPool:
    name = None
    X_i = None
    Y_o = None

    #parameters
    kernel_shape = None
    auto_pad = None
    p = None
    pads = None
    strides = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["kernel_shape", "auto_pad", "p", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LpPool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LpPool'] = LpPool





class Conv:
    name = None
    X_i = None
    W_i = None
    B_i = None
    Y_o = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    input_params = ["X_i", "W_i", "B_i"]
    output_params = ["Y_o"]
    #attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Conv(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Conv'] = Conv





class Not:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Not(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Not'] = Not





class Gather:
    name = None
    data_i = None
    indices_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["data_i", "indices_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Gather(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Gather'] = Gather





class ConvTranspose:
    name = None
    X_i = None
    W_i = None
    B_i = None
    Y_o = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    output_padding = None
    output_shape = None
    pads = None
    strides = None

    input_params = ["X_i", "W_i", "B_i"]
    output_params = ["Y_o"]
    #attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "output_padding", "output_shape", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ConvTranspose(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ConvTranspose'] = ConvTranspose





class Dropout:
    name = None
    data_i = None
    output_o = None
    mask_o = None

    #parameters
    ratio = None

    input_params = ["data_i"]
    output_params = ["output_o", "mask_o"]
    #attribute_params = ["ratio"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Dropout(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Dropout'] = Dropout





class LeakyRelu:
    name = None
    X_i = None
    Y_o = None

    #parameters
    alpha = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LeakyRelu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LeakyRelu'] = LeakyRelu





class Elu:
    name = None
    X_i = None
    Y_o = None

    #parameters
    alpha = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Elu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Elu'] = Elu





class GlobalAveragePool:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._GlobalAveragePool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['GlobalAveragePool'] = GlobalAveragePool





class Gemm:
    name = None
    A_i = None
    B_i = None
    C_i = None
    Y_o = None

    #parameters
    alpha = None
    beta = None
    transA = None
    transB = None

    input_params = ["A_i", "B_i", "C_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha", "beta", "transA", "transB"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Gemm(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Gemm'] = Gemm





class MaxPool:
    name = None
    X_i = None
    Y_o = None
    Indices_o = None

    #parameters
    kernel_shape = None
    auto_pad = None
    ceil_mode = None
    dilations = None
    pads = None
    storage_order = None
    strides = None

    input_params = ["X_i"]
    output_params = ["Y_o", "Indices_o"]
    #attribute_params = ["kernel_shape", "auto_pad", "ceil_mode", "dilations", "pads", "storage_order", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MaxPool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MaxPool'] = MaxPool





class Equal:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Equal(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Equal'] = Equal





class Tile:
    name = None
    input_i = None
    repeats_i = None
    output_o = None

    #parameters

    input_params = ["input_i", "repeats_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Tile(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Tile'] = Tile





class Flatten:
    name = None
    input_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Flatten(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Flatten'] = Flatten





class Floor:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Floor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Floor'] = Floor





class GRU:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_i = None
    W_i = None
    R_i = None
    B_i = None
    sequence_lens_i = None
    initial_h_i = None
    Y_o = None
    Y_h_o = None

    #parameters
    clip = None
    direction = None
    hidden_size = None
    linear_before_reset = None

    input_params = ["X_i", "W_i", "R_i", "B_i", "sequence_lens_i", "initial_h_i"]
    output_params = ["Y_o", "Y_h_o"]
    #attribute_params = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "linear_before_reset"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._GRU(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['GRU'] = GRU





class GlobalLpPool:
    name = None
    X_i = None
    Y_o = None

    #parameters
    p = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["p"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._GlobalLpPool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['GlobalLpPool'] = GlobalLpPool





class Greater:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Greater(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Greater'] = Greater





class HardSigmoid:
    name = None
    X_i = None
    Y_o = None

    #parameters
    alpha = None
    beta = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha", "beta"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._HardSigmoid(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['HardSigmoid'] = HardSigmoid





class Selu:
    name = None
    X_i = None
    Y_o = None

    #parameters
    alpha = None
    gamma = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha", "gamma"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Selu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Selu'] = Selu





class Hardmax:
    name = None
    input_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Hardmax(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Hardmax'] = Hardmax





class If:
    name = None
    cond_i = None

    #parameters
    else_branch = None
    then_branch = None

    input_params = ["cond_i"]
    output_params = []
    #attribute_params = ["else_branch", "then_branch"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._If(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['If'] = If





class Min:
    name = None
    min_o = None

    #parameters

    input_params = []
    output_params = ["min_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Min(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Min'] = Min





class InstanceNormalization:
    name = None
    input_i = None
    scale_i = None
    B_i = None
    output_o = None

    #parameters
    epsilon = None

    input_params = ["input_i", "scale_i", "B_i"]
    output_params = ["output_o"]
    #attribute_params = ["epsilon"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._InstanceNormalization(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['InstanceNormalization'] = InstanceNormalization





class Less:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Less(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Less'] = Less





class EyeLike:
    name = None
    input_i = None
    output_o = None

    #parameters
    dtype = None
    k = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["dtype", "k"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._EyeLike(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['EyeLike'] = EyeLike





class RandomNormal:
    name = None
    output_o = None

    #parameters
    shape = None
    dtype = None
    mean = None
    scale = None
    seed = None

    input_params = []
    output_params = ["output_o"]
    #attribute_params = ["shape", "dtype", "mean", "scale", "seed"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RandomNormal(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RandomNormal'] = RandomNormal





class Slice:
    name = None
    data_i = None
    starts_i = None
    ends_i = None
    axes_i = None
    steps_i = None
    output_o = None

    #parameters

    input_params = ["data_i", "starts_i", "ends_i", "axes_i", "steps_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Slice(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Slice'] = Slice





class PRelu:
    name = None
    X_i = None
    slope_i = None
    Y_o = None

    #parameters

    input_params = ["X_i", "slope_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._PRelu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['PRelu'] = PRelu





class Log:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Log(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Log'] = Log





class LogSoftmax:
    name = None
    input_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LogSoftmax(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LogSoftmax'] = LogSoftmax





class Loop:
    name = None
    M_i = None
    cond_i = None

    #parameters
    body = None

    input_params = ["M_i", "cond_i"]
    output_params = []
    #attribute_params = ["body"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Loop(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Loop'] = Loop





class LpNormalization:
    name = None
    input_i = None
    output_o = None

    #parameters
    axis = None
    p = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis", "p"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LpNormalization(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LpNormalization'] = LpNormalization





class MatMul:
    name = None
    A_i = None
    B_i = None
    Y_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MatMul(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MatMul'] = MatMul





class ReduceL2:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceL2(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceL2'] = ReduceL2





class Max:
    name = None
    max_o = None

    #parameters

    input_params = []
    output_params = ["max_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Max(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Max'] = Max





class MaxRoiPool:
    name = None
    X_i = None
    rois_i = None
    Y_o = None

    #parameters
    pooled_shape = None
    spatial_scale = None

    input_params = ["X_i", "rois_i"]
    output_params = ["Y_o"]
    #attribute_params = ["pooled_shape", "spatial_scale"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MaxRoiPool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MaxRoiPool'] = MaxRoiPool





class Or:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Or(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Or'] = Or





class Pad:
    name = None
    data_i = None
    output_o = None

    #parameters
    pads = None
    mode = None
    value = None

    input_params = ["data_i"]
    output_params = ["output_o"]
    #attribute_params = ["pads", "mode", "value"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Pad(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Pad'] = Pad





class RandomUniformLike:
    name = None
    input_i = None
    output_o = None

    #parameters
    dtype = None
    high = None
    low = None
    seed = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["dtype", "high", "low", "seed"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RandomUniformLike(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RandomUniformLike'] = RandomUniformLike





class Reciprocal:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Reciprocal(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Reciprocal'] = Reciprocal





class Pow:
    name = None
    X_i = None
    Y_i = None
    Z_o = None

    #parameters

    input_params = ["X_i", "Y_i"]
    output_params = ["Z_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Pow(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Pow'] = Pow





class RandomNormalLike:
    name = None
    input_i = None
    output_o = None

    #parameters
    dtype = None
    mean = None
    scale = None
    seed = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["dtype", "mean", "scale", "seed"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RandomNormalLike(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RandomNormalLike'] = RandomNormalLike





class OneHot:
    name = None
    indices_i = None
    depth_i = None
    values_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["indices_i", "depth_i", "values_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._OneHot(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['OneHot'] = OneHot





class RandomUniform:
    name = None
    output_o = None

    #parameters
    shape = None
    dtype = None
    high = None
    low = None
    seed = None

    input_params = []
    output_params = ["output_o"]
    #attribute_params = ["shape", "dtype", "high", "low", "seed"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RandomUniform(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RandomUniform'] = RandomUniform





class ReduceL1:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceL1(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceL1'] = ReduceL1





class ReduceLogSum:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceLogSum(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceLogSum'] = ReduceLogSum





class ReduceLogSumExp:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceLogSumExp(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceLogSumExp'] = ReduceLogSumExp





class ReduceMax:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceMax(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceMax'] = ReduceMax





class OneHotEncoder:
    name = None
    cats_strings = None
    X_i = None
    Y_o = None

    #parameters
    cats_int64s = None
    zeros = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["cats_int64s", "cats_strings", "zeros"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._OneHotEncoder(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['OneHotEncoder'] = OneHotEncoder





class IsNaN:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._IsNaN(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['IsNaN'] = IsNaN





class ReduceMean:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceMean(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceMean'] = ReduceMean





class ReduceMin:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceMin(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceMin'] = ReduceMin





class TreeEnsembleRegressor:
    name = None
    base_values = None
    nodes_hitrates = None
    nodes_modes = None
    nodes_values = None
    target_weights = None
    X_i = None
    Y_o = None

    #parameters
    aggregate_function = None
    n_targets = None
    nodes_falsenodeids = None
    nodes_featureids = None
    nodes_missing_value_tracks_true = None
    nodes_nodeids = None
    nodes_treeids = None
    nodes_truenodeids = None
    post_transform = None
    target_ids = None
    target_nodeids = None
    target_treeids = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["aggregate_function", "base_values", "n_targets", "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates", "nodes_missing_value_tracks_true", "nodes_modes", "nodes_nodeids", "nodes_treeids", "nodes_truenodeids", "nodes_values", "post_transform", "target_ids", "target_nodeids", "target_treeids", "target_weights"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._TreeEnsembleRegressor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['TreeEnsembleRegressor'] = TreeEnsembleRegressor





class ReduceProd:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceProd(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceProd'] = ReduceProd





class ReduceSum:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceSum(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceSum'] = ReduceSum





class ReduceSumSquare:
    name = None
    data_i = None
    reduced_o = None

    #parameters
    axes = None
    keepdims = None

    input_params = ["data_i"]
    output_params = ["reduced_o"]
    #attribute_params = ["axes", "keepdims"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ReduceSumSquare(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ReduceSumSquare'] = ReduceSumSquare





class Relu:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Relu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Relu'] = Relu





class Reshape:
    name = None
    data_i = None
    shape_i = None
    reshaped_o = None

    #parameters

    input_params = ["data_i", "shape_i"]
    output_params = ["reshaped_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Reshape(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Reshape'] = Reshape





class Shape:
    name = None
    data_i = None
    shape_o = None

    #parameters

    input_params = ["data_i"]
    output_params = ["shape_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Shape(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Shape'] = Shape





class Sigmoid:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sigmoid(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sigmoid'] = Sigmoid





class Size:
    name = None
    data_i = None
    size_o = None

    #parameters

    input_params = ["data_i"]
    output_params = ["size_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Size(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Size'] = Size





class Softmax:
    name = None
    input_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Softmax(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Softmax'] = Softmax





class Softplus:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Softplus(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Softplus'] = Softplus





class Softsign:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Softsign(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Softsign'] = Softsign





class SpaceToDepth:
    name = None
    input_i = None
    output_o = None

    #parameters
    blocksize = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["blocksize"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._SpaceToDepth(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['SpaceToDepth'] = SpaceToDepth





class TfIdfVectorizer:
    name = None
    pool_strings = None
    weights = None
    X_i = None
    Y_o = None

    #parameters
    max_gram_length = None
    max_skip_count = None
    min_gram_length = None
    mode = None
    ngram_counts = None
    ngram_indexes = None
    pool_int64s = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["max_gram_length", "max_skip_count", "min_gram_length", "mode", "ngram_counts", "ngram_indexes", "pool_int64s", "pool_strings", "weights"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._TfIdfVectorizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['TfIdfVectorizer'] = TfIdfVectorizer





class Split:
    name = None
    input_i = None

    #parameters
    axis = None
    split = None

    input_params = ["input_i"]
    output_params = []
    #attribute_params = ["axis", "split"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Split(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Split'] = Split





class Imputer:
    name = None
    imputed_value_floats = None
    X_i = None
    Y_o = None

    #parameters
    imputed_value_int64s = None
    replaced_value_float = None
    replaced_value_int64 = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["imputed_value_floats", "imputed_value_int64s", "replaced_value_float", "replaced_value_int64"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Imputer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Imputer'] = Imputer





class Sqrt:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sqrt(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sqrt'] = Sqrt





class Squeeze:
    name = None
    data_i = None
    squeezed_o = None

    #parameters
    axes = None

    input_params = ["data_i"]
    output_params = ["squeezed_o"]
    #attribute_params = ["axes"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Squeeze(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Squeeze'] = Squeeze





class TopK:
    name = None
    X_i = None
    K_i = None
    Values_o = None
    Indices_o = None

    #parameters
    axis = None

    input_params = ["X_i", "K_i"]
    output_params = ["Values_o", "Indices_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._TopK(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['TopK'] = TopK





class Sub:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sub(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sub'] = Sub





class Sum:
    name = None
    sum_o = None

    #parameters

    input_params = []
    output_params = ["sum_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sum(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sum'] = Sum





class Shrink:
    name = None
    input_i = None
    output_o = None

    #parameters
    bias = None
    lambd = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["bias", "lambd"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Shrink(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Shrink'] = Shrink





class Tanh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Tanh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Tanh'] = Tanh





class Transpose:
    name = None
    data_i = None
    transposed_o = None

    #parameters
    perm = None

    input_params = ["data_i"]
    output_params = ["transposed_o"]
    #attribute_params = ["perm"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Transpose(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Transpose'] = Transpose





class Unsqueeze:
    name = None
    data_i = None
    expanded_o = None

    #parameters
    axes = None

    input_params = ["data_i"]
    output_params = ["expanded_o"]
    #attribute_params = ["axes"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Unsqueeze(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Unsqueeze'] = Unsqueeze





class Upsample:
    name = None
    X_i = None
    scales_i = None
    Y_o = None

    #parameters
    mode = None

    input_params = ["X_i", "scales_i"]
    output_params = ["Y_o"]
    #attribute_params = ["mode"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Upsample(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Upsample'] = Upsample





class SVMClassifier:
    name = None
    classlabels_strings = None
    coefficients = None
    kernel_params = None
    prob_a = None
    prob_b = None
    rho = None
    support_vectors = None
    X_i = None
    Y_o = None
    Z_o = None

    #parameters
    classlabels_ints = None
    kernel_type = None
    post_transform = None
    vectors_per_class = None

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    #attribute_params = ["classlabels_ints", "classlabels_strings", "coefficients", "kernel_params", "kernel_type", "post_transform", "prob_a", "prob_b", "rho", "support_vectors", "vectors_per_class"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._SVMClassifier(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['SVMClassifier'] = SVMClassifier





class Xor:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Xor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Xor'] = Xor





class Acos:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Acos(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Acos'] = Acos





class Asin:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Asin(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Asin'] = Asin





class Atan:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Atan(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Atan'] = Atan





class Cos:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Cos(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Cos'] = Cos





class Sin:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sin(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sin'] = Sin





class Tan:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Tan(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Tan'] = Tan





class Multinomial:
    name = None
    input_i = None
    output_o = None

    #parameters
    dtype = None
    sample_size = None
    seed = None

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["dtype", "sample_size", "seed"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Multinomial(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Multinomial'] = Multinomial





class Scan:
    name = None

    #parameters
    body = None
    num_scan_inputs = None
    scan_input_axes = None
    scan_input_directions = None
    scan_output_axes = None
    scan_output_directions = None

    input_params = []
    output_params = []
    #attribute_params = ["body", "num_scan_inputs", "scan_input_axes", "scan_input_directions", "scan_output_axes", "scan_output_directions"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Scan(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Scan'] = Scan





class Compress:
    name = None
    input_i = None
    condition_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["input_i", "condition_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Compress(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Compress'] = Compress





class ConstantOfShape:
    name = None
    value = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = ["value"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ConstantOfShape(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ConstantOfShape'] = ConstantOfShape





class MaxUnpool:
    name = None
    X_i = None
    I_i = None
    output_shape_i = None
    output_o = None

    #parameters
    kernel_shape = None
    pads = None
    strides = None

    input_params = ["X_i", "I_i", "output_shape_i"]
    output_params = ["output_o"]
    #attribute_params = ["kernel_shape", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MaxUnpool(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MaxUnpool'] = MaxUnpool





class Scatter:
    name = None
    data_i = None
    indices_i = None
    updates_i = None
    output_o = None

    #parameters
    axis = None

    input_params = ["data_i", "indices_i", "updates_i"]
    output_params = ["output_o"]
    #attribute_params = ["axis"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Scatter(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Scatter'] = Scatter





class Sinh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sinh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sinh'] = Sinh





class Cosh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Cosh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Cosh'] = Cosh





class Asinh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Asinh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Asinh'] = Asinh





class Acosh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Acosh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Acosh'] = Acosh





class NonMaxSuppression:
    name = None
    boxes_i = None
    scores_i = None
    max_output_boxes_per_class_i = None
    iou_threshold_i = None
    score_threshold_i = None
    selected_indices_o = None

    #parameters
    center_point_box = None

    input_params = ["boxes_i", "scores_i", "max_output_boxes_per_class_i", "iou_threshold_i", "score_threshold_i"]
    output_params = ["selected_indices_o"]
    #attribute_params = ["center_point_box"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._NonMaxSuppression(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['NonMaxSuppression'] = NonMaxSuppression





class Atanh:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Atanh(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Atanh'] = Atanh





class Sign:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Sign(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Sign'] = Sign





class Erf:
    name = None
    input_i = None
    output_o = None

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Erf(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Erf'] = Erf





class Where:
    name = None
    condition_i = None
    X_i = None
    Y_i = None
    output_o = None

    #parameters

    input_params = ["condition_i", "X_i", "Y_i"]
    output_params = ["output_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Where(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Where'] = Where





class NonZero:
    name = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._NonZero(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['NonZero'] = NonZero





class MeanVarianceNormalization:
    name = None
    X_i = None
    Y_o = None

    #parameters
    axes = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["axes"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MeanVarianceNormalization(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MeanVarianceNormalization'] = MeanVarianceNormalization





class StringNormalizer:
    name = None
    stopwords = None
    X_i = None
    Y_o = None

    #parameters
    case_change_action = None
    is_case_sensitive = None
    locale = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["case_change_action", "is_case_sensitive", "locale", "stopwords"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._StringNormalizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['StringNormalizer'] = StringNormalizer





class Mod:
    name = None
    A_i = None
    B_i = None
    C_o = None

    #parameters
    fmod = None

    input_params = ["A_i", "B_i"]
    output_params = ["C_o"]
    #attribute_params = ["fmod"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Mod(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Mod'] = Mod





class ThresholdedRelu:
    name = None
    X_i = None
    Y_o = None

    #parameters
    alpha = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["alpha"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ThresholdedRelu(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ThresholdedRelu'] = ThresholdedRelu





class MatMulInteger:
    name = None
    A_i = None
    B_i = None
    a_zero_point_i = None
    b_zero_point_i = None
    Y_o = None

    #parameters

    input_params = ["A_i", "B_i", "a_zero_point_i", "b_zero_point_i"]
    output_params = ["Y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._MatMulInteger(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['MatMulInteger'] = MatMulInteger





class QLinearMatMul:
    name = None
    a_i = None
    a_scale_i = None
    a_zero_point_i = None
    b_i = None
    b_scale_i = None
    b_zero_point_i = None
    y_scale_i = None
    y_zero_point_i = None
    y_o = None

    #parameters

    input_params = ["a_i", "a_scale_i", "a_zero_point_i", "b_i", "b_scale_i", "b_zero_point_i", "y_scale_i", "y_zero_point_i"]
    output_params = ["y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._QLinearMatMul(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['QLinearMatMul'] = QLinearMatMul





class ConvInteger:
    name = None
    x_i = None
    w_i = None
    x_zero_point_i = None
    w_zero_point_i = None
    y_o = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    input_params = ["x_i", "w_i", "x_zero_point_i", "w_zero_point_i"]
    output_params = ["y_o"]
    #attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ConvInteger(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ConvInteger'] = ConvInteger





class QLinearConv:
    name = None
    x_i = None
    x_scale_i = None
    x_zero_point_i = None
    w_i = None
    w_scale_i = None
    w_zero_point_i = None
    y_scale_i = None
    y_zero_point_i = None
    B_i = None
    y_o = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    input_params = ["x_i", "x_scale_i", "x_zero_point_i", "w_i", "w_scale_i", "w_zero_point_i", "y_scale_i", "y_zero_point_i", "B_i"]
    output_params = ["y_o"]
    #attribute_params = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._QLinearConv(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['QLinearConv'] = QLinearConv





class QuantizeLinear:
    name = None
    x_i = None
    y_scale_i = None
    y_zero_point_i = None
    y_o = None

    #parameters

    input_params = ["x_i", "y_scale_i", "y_zero_point_i"]
    output_params = ["y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._QuantizeLinear(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['QuantizeLinear'] = QuantizeLinear





class DequantizeLinear:
    name = None
    x_i = None
    x_scale_i = None
    x_zero_point_i = None
    y_o = None

    #parameters

    input_params = ["x_i", "x_scale_i", "x_zero_point_i"]
    output_params = ["y_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._DequantizeLinear(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['DequantizeLinear'] = DequantizeLinear





class IsInf:
    name = None
    X_i = None
    Y_o = None

    #parameters
    detect_negative = None
    detect_positive = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["detect_negative", "detect_positive"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._IsInf(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['IsInf'] = IsInf





class RoiAlign:
    name = None
    X_i = None
    rois_i = None
    batch_indices_i = None
    Y_o = None

    #parameters
    mode = None
    output_height = None
    output_width = None
    sampling_ratio = None
    spatial_scale = None

    input_params = ["X_i", "rois_i", "batch_indices_i"]
    output_params = ["Y_o"]
    #attribute_params = ["mode", "output_height", "output_width", "sampling_ratio", "spatial_scale"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._RoiAlign(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['RoiAlign'] = RoiAlign





class ArrayFeatureExtractor:
    name = None
    X_i = None
    Y_i = None
    Z_o = None

    #parameters

    input_params = ["X_i", "Y_i"]
    output_params = ["Z_o"]
    #attribute_params = []

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ArrayFeatureExtractor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ArrayFeatureExtractor'] = ArrayFeatureExtractor





class Binarizer:
    name = None
    X_i = None
    Y_o = None

    #parameters
    threshold = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["threshold"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Binarizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Binarizer'] = Binarizer





class CategoryMapper:
    name = None
    cats_strings = None
    X_i = None
    Y_o = None

    #parameters
    cats_int64s = None
    default_int64 = None
    default_string = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["cats_int64s", "cats_strings", "default_int64", "default_string"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._CategoryMapper(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['CategoryMapper'] = CategoryMapper





class DictVectorizer:
    name = None
    string_vocabulary = None
    X_i = None
    Y_o = None

    #parameters
    int64_vocabulary = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["int64_vocabulary", "string_vocabulary"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._DictVectorizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['DictVectorizer'] = DictVectorizer





class FeatureVectorizer:
    name = None
    Y_o = None

    #parameters
    inputdimensions = None

    input_params = []
    output_params = ["Y_o"]
    #attribute_params = ["inputdimensions"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._FeatureVectorizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['FeatureVectorizer'] = FeatureVectorizer





class LabelEncoder:
    name = None
    keys_floats = None
    keys_strings = None
    values_floats = None
    values_strings = None
    X_i = None
    Y_o = None

    #parameters
    default_float = None
    default_int64 = None
    default_string = None
    keys_int64s = None
    values_int64s = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["default_float", "default_int64", "default_string", "keys_floats", "keys_int64s", "keys_strings", "values_floats", "values_int64s", "values_strings"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LabelEncoder(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LabelEncoder'] = LabelEncoder





class LinearClassifier:
    name = None
    coefficients = None
    classlabels_strings = None
    intercepts = None
    X_i = None
    Y_o = None
    Z_o = None

    #parameters
    classlabels_ints = None
    multi_class = None
    post_transform = None

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    #attribute_params = ["coefficients", "classlabels_ints", "classlabels_strings", "intercepts", "multi_class", "post_transform"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LinearClassifier(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LinearClassifier'] = LinearClassifier





class LinearRegressor:
    name = None
    coefficients = None
    intercepts = None
    X_i = None
    Y_o = None

    #parameters
    post_transform = None
    targets = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["coefficients", "intercepts", "post_transform", "targets"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._LinearRegressor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['LinearRegressor'] = LinearRegressor





class Normalizer:
    name = None
    X_i = None
    Y_o = None

    #parameters
    norm = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["norm"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Normalizer(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Normalizer'] = Normalizer





class SVMRegressor:
    name = None
    coefficients = None
    kernel_params = None
    rho = None
    support_vectors = None
    X_i = None
    Y_o = None

    #parameters
    kernel_type = None
    n_supports = None
    one_class = None
    post_transform = None

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["coefficients", "kernel_params", "kernel_type", "n_supports", "one_class", "post_transform", "rho", "support_vectors"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._SVMRegressor(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['SVMRegressor'] = SVMRegressor





class Scaler:
    name = None
    offset = None
    scale = None
    X_i = None
    Y_o = None

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    #attribute_params = ["offset", "scale"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._Scaler(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['Scaler'] = Scaler





class TreeEnsembleClassifier:
    name = None
    base_values = None
    class_weights = None
    classlabels_strings = None
    nodes_hitrates = None
    nodes_modes = None
    nodes_values = None
    X_i = None
    Y_o = None
    Z_o = None

    #parameters
    class_ids = None
    class_nodeids = None
    class_treeids = None
    classlabels_int64s = None
    nodes_falsenodeids = None
    nodes_featureids = None
    nodes_missing_value_tracks_true = None
    nodes_nodeids = None
    nodes_treeids = None
    nodes_truenodeids = None
    post_transform = None

    input_params = ["X_i"]
    output_params = ["Y_o", "Z_o"]
    #attribute_params = ["base_values", "class_ids", "class_nodeids", "class_treeids", "class_weights", "classlabels_int64s", "classlabels_strings", "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates", "nodes_missing_value_tracks_true", "nodes_modes", "nodes_nodeids", "nodes_treeids", "nodes_truenodeids", "nodes_values", "post_transform"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._TreeEnsembleClassifier(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['TreeEnsembleClassifier'] = TreeEnsembleClassifier





class ZipMap:
    name = None
    classlabels_strings = None
    X_i = None
    Z_o = None

    #parameters
    classlabels_int64s = None

    input_params = ["X_i"]
    output_params = ["Z_o"]
    #attribute_params = ["classlabels_int64s", "classlabels_strings"]

    def __init__(self, name):
        self.name = name
        #self.Module = nn._ZipMap(name)

    def input(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x

    def output(self, *args):        
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x
    
    def attribute(self, **kwargs):
        self.__dict__.update(kwargs)

    def call(self):
        pass

layer_map['ZipMap'] = ZipMap

