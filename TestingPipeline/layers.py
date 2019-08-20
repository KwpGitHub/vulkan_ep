import _backend.nn as nn
layer_map = {}


class LSTM:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_input = None
    W_input = None
    R_input = None
    B_input_opt = None
    sequence_lens_input_opt = None
    initial_h_input_opt = None
    initial_c_input_opt = None
    P_input_opt = None
    Y_output_opt = None
    Y_h_output_opt = None
    Y_c_output_opt = None

    #parameters
    clip = None
    direction = None
    hidden_size = None
    input_forget = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LSTM(name)
    def input(self, *args):
        inpts = ["X_input", "W_input", "R_input", "B_input_opt", "sequence_lens_input_opt", "initial_h_input_opt", "initial_c_input_opt", "P_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output_opt", "Y_h_output_opt", "Y_c_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['lstm'] = LSTM





class Identity:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Identity(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['identity'] = Identity





class Abs:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Abs(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['abs'] = Abs





class BatchNormalization:
    name = None
    X_input = None
    scale_input = None
    B_input = None
    mean_input = None
    var_input = None
    Y_output = None
    mean_output_opt = None
    var_output_opt = None
    saved_mean_output_opt = None
    saved_var_output_opt = None

    #parameters
    epsilon = None
    momentum = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._BatchNormalization(name)
    def input(self, *args):
        inpts = ["X_input", "scale_input", "B_input", "mean_input", "var_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output", "mean_output_opt", "var_output_opt", "saved_mean_output_opt", "saved_var_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['batchnormalization'] = BatchNormalization





class Mean:
    name = None
    mean_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Mean(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["mean_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['mean'] = Mean





class Add:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Add(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['add'] = Add





class GlobalMaxPool:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalMaxPool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['globalmaxpool'] = GlobalMaxPool





class Cast:
    name = None
    input_input = None
    output_output = None

    #parameters
    to = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Cast(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['cast'] = Cast





class AveragePool:
    name = None
    X_input = None
    Y_output = None

    #parameters
    kernel_shape = None
    auto_pad = None
    ceil_mode = None
    count_include_pad = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._AveragePool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['averagepool'] = AveragePool





class And:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._And(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['and'] = And





class LRN:
    name = None
    X_input = None
    Y_output = None

    #parameters
    size = None
    alpha = None
    beta = None
    bias = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LRN(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['lrn'] = LRN





class ArgMax:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axis = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ArgMax(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['argmax'] = ArgMax





class Resize:
    name = None
    X_input = None
    scales_input = None
    Y_output = None

    #parameters
    mode = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Resize(name)
    def input(self, *args):
        inpts = ["X_input", "scales_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['resize'] = Resize





class Expand:
    name = None
    input_input = None
    shape_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Expand(name)
    def input(self, *args):
        inpts = ["input_input", "shape_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['expand'] = Expand





class Neg:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Neg(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['neg'] = Neg





class Mul:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Mul(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['mul'] = Mul





class ArgMin:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axis = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ArgMin(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['argmin'] = ArgMin





class CastMap:
    name = None
    X_input = None
    Y_output = None

    #parameters
    cast_to = None
    map_form = None
    max_map = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._CastMap(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['castmap'] = CastMap





class Exp:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Exp(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['exp'] = Exp





class Div:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Div(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['div'] = Div





class ReverseSequence:
    name = None
    input_input = None
    sequence_lens_input = None
    Y_output = None

    #parameters
    batch_axis = None
    time_axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReverseSequence(name)
    def input(self, *args):
        inpts = ["input_input", "sequence_lens_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reversesequence'] = ReverseSequence





class Ceil:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Ceil(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['ceil'] = Ceil





class DepthToSpace:
    name = None
    input_input = None
    output_output = None

    #parameters
    blocksize = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._DepthToSpace(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['depthtospace'] = DepthToSpace





class Clip:
    name = None
    input_input = None
    output_output = None

    #parameters
    max = None
    min = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Clip(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['clip'] = Clip





class RNN:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_input = None
    W_input = None
    R_input = None
    B_input_opt = None
    sequence_lens_input_opt = None
    initial_h_input_opt = None
    Y_output_opt = None
    Y_h_output_opt = None

    #parameters
    clip = None
    direction = None
    hidden_size = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RNN(name)
    def input(self, *args):
        inpts = ["X_input", "W_input", "R_input", "B_input_opt", "sequence_lens_input_opt", "initial_h_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output_opt", "Y_h_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['rnn'] = RNN





class Concat:
    name = None
    concat_result_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Concat(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["concat_result_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['concat'] = Concat





class Constant:
    name = None
    value = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Constant(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['constant'] = Constant





class LpPool:
    name = None
    X_input = None
    Y_output = None

    #parameters
    kernel_shape = None
    auto_pad = None
    p = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LpPool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['lppool'] = LpPool





class Conv:
    name = None
    X_input = None
    W_input = None
    B_input_opt = None
    Y_output = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Conv(name)
    def input(self, *args):
        inpts = ["X_input", "W_input", "B_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['conv'] = Conv





class Not:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Not(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['not'] = Not





class Gather:
    name = None
    data_input = None
    indices_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Gather(name)
    def input(self, *args):
        inpts = ["data_input", "indices_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['gather'] = Gather





class ConvTranspose:
    name = None
    X_input = None
    W_input = None
    B_input_opt = None
    Y_output = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    output_padding = None
    output_shape = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ConvTranspose(name)
    def input(self, *args):
        inpts = ["X_input", "W_input", "B_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['convtranspose'] = ConvTranspose





class Dropout:
    name = None
    data_input = None
    output_output = None
    mask_output_opt = None

    #parameters
    ratio = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Dropout(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output", "mask_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['dropout'] = Dropout





class LeakyRelu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LeakyRelu(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['leakyrelu'] = LeakyRelu





class Elu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Elu(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['elu'] = Elu





class GlobalAveragePool:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalAveragePool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['globalaveragepool'] = GlobalAveragePool





class Gemm:
    name = None
    A_input = None
    B_input = None
    C_input = None
    Y_output = None

    #parameters
    alpha = None
    beta = None
    transA = None
    transB = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Gemm(name)
    def input(self, *args):
        inpts = ["A_input", "B_input", "C_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['gemm'] = Gemm





class MaxPool:
    name = None
    X_input = None
    Y_output = None
    Indices_output_opt = None

    #parameters
    kernel_shape = None
    auto_pad = None
    ceil_mode = None
    dilations = None
    pads = None
    storage_order = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxPool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output", "Indices_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['maxpool'] = MaxPool





class Equal:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Equal(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['equal'] = Equal





class Tile:
    name = None
    input_input = None
    repeats_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Tile(name)
    def input(self, *args):
        inpts = ["input_input", "repeats_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['tile'] = Tile





class Flatten:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Flatten(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['flatten'] = Flatten





class Floor:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Floor(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['floor'] = Floor





class GRU:
    name = None
    activation_alpha = None
    activation_beta = None
    activations = None
    X_input = None
    W_input = None
    R_input = None
    B_input_opt = None
    sequence_lens_input_opt = None
    initial_h_input_opt = None
    Y_output_opt = None
    Y_h_output_opt = None

    #parameters
    clip = None
    direction = None
    hidden_size = None
    linear_before_reset = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._GRU(name)
    def input(self, *args):
        inpts = ["X_input", "W_input", "R_input", "B_input_opt", "sequence_lens_input_opt", "initial_h_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output_opt", "Y_h_output_opt"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['gru'] = GRU





class GlobalLpPool:
    name = None
    X_input = None
    Y_output = None

    #parameters
    p = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._GlobalLpPool(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['globallppool'] = GlobalLpPool





class Greater:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Greater(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['greater'] = Greater





class HardSigmoid:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None
    beta = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._HardSigmoid(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['hardsigmoid'] = HardSigmoid





class Selu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None
    gamma = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Selu(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['selu'] = Selu





class Hardmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Hardmax(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['hardmax'] = Hardmax





class If:
    name = None
    cond_input = None

    #parameters
    else_branch = None
    then_branch = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._If(name)
    def input(self, *args):
        inpts = ["cond_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = []
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['if'] = If





class Min:
    name = None
    min_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Min(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["min_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['min'] = Min





class InstanceNormalization:
    name = None
    input_input = None
    scale_input = None
    B_input = None
    output_output = None

    #parameters
    epsilon = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._InstanceNormalization(name)
    def input(self, *args):
        inpts = ["input_input", "scale_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['instancenormalization'] = InstanceNormalization





class Less:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Less(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['less'] = Less





class EyeLike:
    name = None
    input_input = None
    output_output = None

    #parameters
    dtype = None
    k = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._EyeLike(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['eyelike'] = EyeLike





class RandomNormal:
    name = None
    output_output = None

    #parameters
    shape = None
    dtype = None
    mean = None
    scale = None
    seed = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomNormal(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['randomnormal'] = RandomNormal





class Slice:
    name = None
    data_input = None
    starts_input = None
    ends_input = None
    axes_input_opt = None
    steps_input_opt = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Slice(name)
    def input(self, *args):
        inpts = ["data_input", "starts_input", "ends_input", "axes_input_opt", "steps_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['slice'] = Slice





class PRelu:
    name = None
    X_input = None
    slope_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._PRelu(name)
    def input(self, *args):
        inpts = ["X_input", "slope_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['prelu'] = PRelu





class Log:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Log(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['log'] = Log





class LogSoftmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LogSoftmax(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['logsoftmax'] = LogSoftmax





class Loop:
    name = None
    M_input_opt = None
    cond_input_opt = None

    #parameters
    body = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Loop(name)
    def input(self, *args):
        inpts = ["M_input_opt", "cond_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = []
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['loop'] = Loop





class LpNormalization:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None
    p = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LpNormalization(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['lpnormalization'] = LpNormalization





class MatMul:
    name = None
    A_input = None
    B_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._MatMul(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['matmul'] = MatMul





class ReduceL2:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceL2(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducel2'] = ReduceL2





class Max:
    name = None
    max_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Max(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["max_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['max'] = Max





class MaxRoiPool:
    name = None
    X_input = None
    rois_input = None
    Y_output = None

    #parameters
    pooled_shape = None
    spatial_scale = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxRoiPool(name)
    def input(self, *args):
        inpts = ["X_input", "rois_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['maxroipool'] = MaxRoiPool





class Or:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Or(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['or'] = Or





class Pad:
    name = None
    data_input = None
    output_output = None

    #parameters
    pads = None
    mode = None
    value = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Pad(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['pad'] = Pad





class RandomUniformLike:
    name = None
    input_input = None
    output_output = None

    #parameters
    dtype = None
    high = None
    low = None
    seed = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomUniformLike(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['randomuniformlike'] = RandomUniformLike





class Reciprocal:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Reciprocal(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reciprocal'] = Reciprocal





class Pow:
    name = None
    X_input = None
    Y_input = None
    Z_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Pow(name)
    def input(self, *args):
        inpts = ["X_input", "Y_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['pow'] = Pow





class RandomNormalLike:
    name = None
    input_input = None
    output_output = None

    #parameters
    dtype = None
    mean = None
    scale = None
    seed = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomNormalLike(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['randomnormallike'] = RandomNormalLike





class OneHot:
    name = None
    indices_input = None
    depth_input = None
    values_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._OneHot(name)
    def input(self, *args):
        inpts = ["indices_input", "depth_input", "values_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['onehot'] = OneHot





class RandomUniform:
    name = None
    output_output = None

    #parameters
    shape = None
    dtype = None
    high = None
    low = None
    seed = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RandomUniform(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['randomuniform'] = RandomUniform





class ReduceL1:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceL1(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducel1'] = ReduceL1





class ReduceLogSum:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceLogSum(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducelogsum'] = ReduceLogSum





class ReduceLogSumExp:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceLogSumExp(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducelogsumexp'] = ReduceLogSumExp





class ReduceMax:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMax(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducemax'] = ReduceMax





class OneHotEncoder:
    name = None
    cats_strings = None
    X_input = None
    Y_output = None

    #parameters
    cats_int64s = None
    zeros = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._OneHotEncoder(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['onehotencoder'] = OneHotEncoder





class IsNaN:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._IsNaN(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['isnan'] = IsNaN





class ReduceMean:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMean(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducemean'] = ReduceMean





class ReduceMin:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceMin(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducemin'] = ReduceMin





class TreeEnsembleRegressor:
    name = None
    base_values = None
    nodes_hitrates = None
    nodes_modes = None
    nodes_values = None
    target_weights = None
    X_input = None
    Y_output = None

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

    def __init__(self, name):
        self.name = name
        self.Module = nn._TreeEnsembleRegressor(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['treeensembleregressor'] = TreeEnsembleRegressor





class ReduceProd:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceProd(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reduceprod'] = ReduceProd





class ReduceSum:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceSum(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducesum'] = ReduceSum





class ReduceSumSquare:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ReduceSumSquare(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reduced_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reducesumsquare'] = ReduceSumSquare





class Relu:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Relu(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['relu'] = Relu





class Reshape:
    name = None
    data_input = None
    shape_input = None
    reshaped_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Reshape(name)
    def input(self, *args):
        inpts = ["data_input", "shape_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["reshaped_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['reshape'] = Reshape





class Shape:
    name = None
    data_input = None
    shape_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Shape(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["shape_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['shape'] = Shape





class Sigmoid:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sigmoid(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sigmoid'] = Sigmoid





class Size:
    name = None
    data_input = None
    size_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Size(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["size_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['size'] = Size





class Softmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Softmax(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['softmax'] = Softmax





class Softplus:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Softplus(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['softplus'] = Softplus





class Softsign:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Softsign(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['softsign'] = Softsign





class SpaceToDepth:
    name = None
    input_input = None
    output_output = None

    #parameters
    blocksize = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._SpaceToDepth(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['spacetodepth'] = SpaceToDepth





class TfIdfVectorizer:
    name = None
    pool_strings = None
    weights = None
    X_input = None
    Y_output = None

    #parameters
    max_gram_length = None
    max_skip_count = None
    min_gram_length = None
    mode = None
    ngram_counts = None
    ngram_indexes = None
    pool_int64s = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._TfIdfVectorizer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['tfidfvectorizer'] = TfIdfVectorizer





class Split:
    name = None
    input_input = None

    #parameters
    axis = None
    split = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Split(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = []
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['split'] = Split





class Imputer:
    name = None
    imputed_value_floats = None
    X_input = None
    Y_output = None

    #parameters
    imputed_value_int64s = None
    replaced_value_float = None
    replaced_value_int64 = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Imputer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['imputer'] = Imputer





class Sqrt:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sqrt(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sqrt'] = Sqrt





class Squeeze:
    name = None
    data_input = None
    squeezed_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Squeeze(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["squeezed_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['squeeze'] = Squeeze





class TopK:
    name = None
    X_input = None
    K_input = None
    Values_output = None
    Indices_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._TopK(name)
    def input(self, *args):
        inpts = ["X_input", "K_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Values_output", "Indices_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['topk'] = TopK





class Sub:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sub(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sub'] = Sub





class Sum:
    name = None
    sum_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sum(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["sum_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sum'] = Sum





class Shrink:
    name = None
    input_input = None
    output_output = None

    #parameters
    bias = None
    lambd = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Shrink(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['shrink'] = Shrink





class Tanh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Tanh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['tanh'] = Tanh





class Transpose:
    name = None
    data_input = None
    transposed_output = None

    #parameters
    perm = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Transpose(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["transposed_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['transpose'] = Transpose





class Unsqueeze:
    name = None
    data_input = None
    expanded_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Unsqueeze(name)
    def input(self, *args):
        inpts = ["data_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["expanded_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['unsqueeze'] = Unsqueeze





class Upsample:
    name = None
    X_input = None
    scales_input = None
    Y_output = None

    #parameters
    mode = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Upsample(name)
    def input(self, *args):
        inpts = ["X_input", "scales_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['upsample'] = Upsample





class SVMClassifier:
    name = None
    classlabels_strings = None
    coefficients = None
    kernel_params = None
    prob_a = None
    prob_b = None
    rho = None
    support_vectors = None
    X_input = None
    Y_output = None
    Z_output = None

    #parameters
    classlabels_ints = None
    kernel_type = None
    post_transform = None
    vectors_per_class = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._SVMClassifier(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output", "Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['svmclassifier'] = SVMClassifier





class Xor:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Xor(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['xor'] = Xor





class Acos:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Acos(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['acos'] = Acos





class Asin:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Asin(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['asin'] = Asin





class Atan:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Atan(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['atan'] = Atan





class Cos:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Cos(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['cos'] = Cos





class Sin:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sin(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sin'] = Sin





class Tan:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Tan(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['tan'] = Tan





class Multinomial:
    name = None
    input_input = None
    output_output = None

    #parameters
    dtype = None
    sample_size = None
    seed = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Multinomial(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['multinomial'] = Multinomial





class Scan:
    name = None

    #parameters
    body = None
    num_scan_inputs = None
    scan_input_axes = None
    scan_input_directions = None
    scan_output_axes = None
    scan_output_directions = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Scan(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = []
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['scan'] = Scan





class Compress:
    name = None
    input_input = None
    condition_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Compress(name)
    def input(self, *args):
        inpts = ["input_input", "condition_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['compress'] = Compress





class ConstantOfShape:
    name = None
    value = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._ConstantOfShape(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['constantofshape'] = ConstantOfShape





class MaxUnpool:
    name = None
    X_input = None
    I_input = None
    output_shape_input_opt = None
    output_output = None

    #parameters
    kernel_shape = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._MaxUnpool(name)
    def input(self, *args):
        inpts = ["X_input", "I_input", "output_shape_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['maxunpool'] = MaxUnpool





class Scatter:
    name = None
    data_input = None
    indices_input = None
    updates_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Scatter(name)
    def input(self, *args):
        inpts = ["data_input", "indices_input", "updates_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['scatter'] = Scatter





class Sinh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sinh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sinh'] = Sinh





class Cosh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Cosh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['cosh'] = Cosh





class Asinh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Asinh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['asinh'] = Asinh





class Acosh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Acosh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['acosh'] = Acosh





class NonMaxSuppression:
    name = None
    boxes_input = None
    scores_input = None
    max_output_boxes_per_class_input_opt = None
    iou_threshold_input_opt = None
    score_threshold_input_opt = None
    selected_indices_output = None

    #parameters
    center_point_box = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._NonMaxSuppression(name)
    def input(self, *args):
        inpts = ["boxes_input", "scores_input", "max_output_boxes_per_class_input_opt", "iou_threshold_input_opt", "score_threshold_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["selected_indices_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['nonmaxsuppression'] = NonMaxSuppression





class Atanh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Atanh(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['atanh'] = Atanh





class Sign:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Sign(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['sign'] = Sign





class Erf:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Erf(name)
    def input(self, *args):
        inpts = ["input_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['erf'] = Erf





class Where:
    name = None
    condition_input = None
    X_input = None
    Y_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Where(name)
    def input(self, *args):
        inpts = ["condition_input", "X_input", "Y_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["output_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['where'] = Where





class NonZero:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._NonZero(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['nonzero'] = NonZero





class MeanVarianceNormalization:
    name = None
    X_input = None
    Y_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._MeanVarianceNormalization(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['meanvariancenormalization'] = MeanVarianceNormalization





class StringNormalizer:
    name = None
    stopwords = None
    X_input = None
    Y_output = None

    #parameters
    case_change_action = None
    is_case_sensitive = None
    locale = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._StringNormalizer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['stringnormalizer'] = StringNormalizer





class Mod:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters
    fmod = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Mod(name)
    def input(self, *args):
        inpts = ["A_input", "B_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["C_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['mod'] = Mod





class ThresholdedRelu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ThresholdedRelu(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['thresholdedrelu'] = ThresholdedRelu





class MatMulInteger:
    name = None
    A_input = None
    B_input = None
    a_zero_point_input_opt = None
    b_zero_point_input_opt = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._MatMulInteger(name)
    def input(self, *args):
        inpts = ["A_input", "B_input", "a_zero_point_input_opt", "b_zero_point_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['matmulinteger'] = MatMulInteger





class QLinearMatMul:
    name = None
    a_input = None
    a_scale_input = None
    a_zero_point_input = None
    b_input = None
    b_scale_input = None
    b_zero_point_input = None
    y_scale_input = None
    y_zero_point_input = None
    y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._QLinearMatMul(name)
    def input(self, *args):
        inpts = ["a_input", "a_scale_input", "a_zero_point_input", "b_input", "b_scale_input", "b_zero_point_input", "y_scale_input", "y_zero_point_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['qlinearmatmul'] = QLinearMatMul





class ConvInteger:
    name = None
    x_input = None
    w_input = None
    x_zero_point_input_opt = None
    w_zero_point_input_opt = None
    y_output = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ConvInteger(name)
    def input(self, *args):
        inpts = ["x_input", "w_input", "x_zero_point_input_opt", "w_zero_point_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['convinteger'] = ConvInteger





class QLinearConv:
    name = None
    x_input = None
    x_scale_input = None
    x_zero_point_input = None
    w_input = None
    w_scale_input = None
    w_zero_point_input = None
    y_scale_input = None
    y_zero_point_input = None
    B_input_opt = None
    y_output = None

    #parameters
    auto_pad = None
    dilations = None
    group = None
    kernel_shape = None
    pads = None
    strides = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._QLinearConv(name)
    def input(self, *args):
        inpts = ["x_input", "x_scale_input", "x_zero_point_input", "w_input", "w_scale_input", "w_zero_point_input", "y_scale_input", "y_zero_point_input", "B_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['qlinearconv'] = QLinearConv





class QuantizeLinear:
    name = None
    x_input = None
    y_scale_input = None
    y_zero_point_input_opt = None
    y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._QuantizeLinear(name)
    def input(self, *args):
        inpts = ["x_input", "y_scale_input", "y_zero_point_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['quantizelinear'] = QuantizeLinear





class DequantizeLinear:
    name = None
    x_input = None
    x_scale_input = None
    x_zero_point_input_opt = None
    y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._DequantizeLinear(name)
    def input(self, *args):
        inpts = ["x_input", "x_scale_input", "x_zero_point_input_opt"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['dequantizelinear'] = DequantizeLinear





class IsInf:
    name = None
    X_input = None
    Y_output = None

    #parameters
    detect_negative = None
    detect_positive = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._IsInf(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['isinf'] = IsInf





class RoiAlign:
    name = None
    X_input = None
    rois_input = None
    batch_indices_input = None
    Y_output = None

    #parameters
    mode = None
    output_height = None
    output_width = None
    sampling_ratio = None
    spatial_scale = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._RoiAlign(name)
    def input(self, *args):
        inpts = ["X_input", "rois_input", "batch_indices_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['roialign'] = RoiAlign





class ArrayFeatureExtractor:
    name = None
    X_input = None
    Y_input = None
    Z_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._ArrayFeatureExtractor(name)
    def input(self, *args):
        inpts = ["X_input", "Y_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['arrayfeatureextractor'] = ArrayFeatureExtractor





class Binarizer:
    name = None
    X_input = None
    Y_output = None

    #parameters
    threshold = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Binarizer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['binarizer'] = Binarizer





class CategoryMapper:
    name = None
    cats_strings = None
    X_input = None
    Y_output = None

    #parameters
    cats_int64s = None
    default_int64 = None
    default_string = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._CategoryMapper(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['categorymapper'] = CategoryMapper





class DictVectorizer:
    name = None
    string_vocabulary = None
    X_input = None
    Y_output = None

    #parameters
    int64_vocabulary = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._DictVectorizer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['dictvectorizer'] = DictVectorizer





class FeatureVectorizer:
    name = None
    Y_output = None

    #parameters
    inputdimensions = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._FeatureVectorizer(name)
    def input(self, *args):
        inpts = []
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['featurevectorizer'] = FeatureVectorizer





class LabelEncoder:
    name = None
    keys_floats = None
    keys_strings = None
    values_floats = None
    values_strings = None
    X_input = None
    Y_output = None

    #parameters
    default_float = None
    default_int64 = None
    default_string = None
    keys_int64s = None
    values_int64s = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LabelEncoder(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['labelencoder'] = LabelEncoder





class LinearClassifier:
    name = None
    coefficients = None
    classlabels_strings = None
    intercepts = None
    X_input = None
    Y_output = None
    Z_output = None

    #parameters
    classlabels_ints = None
    multi_class = None
    post_transform = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LinearClassifier(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output", "Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['linearclassifier'] = LinearClassifier





class LinearRegressor:
    name = None
    coefficients = None
    intercepts = None
    X_input = None
    Y_output = None

    #parameters
    post_transform = None
    targets = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._LinearRegressor(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['linearregressor'] = LinearRegressor





class Normalizer:
    name = None
    X_input = None
    Y_output = None

    #parameters
    norm = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._Normalizer(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['normalizer'] = Normalizer





class SVMRegressor:
    name = None
    coefficients = None
    kernel_params = None
    rho = None
    support_vectors = None
    X_input = None
    Y_output = None

    #parameters
    kernel_type = None
    n_supports = None
    one_class = None
    post_transform = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._SVMRegressor(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['svmregressor'] = SVMRegressor





class Scaler:
    name = None
    offset = None
    scale = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name
        self.Module = nn._Scaler(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['scaler'] = Scaler





class TreeEnsembleClassifier:
    name = None
    base_values = None
    class_weights = None
    classlabels_strings = None
    nodes_hitrates = None
    nodes_modes = None
    nodes_values = None
    X_input = None
    Y_output = None
    Z_output = None

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

    def __init__(self, name):
        self.name = name
        self.Module = nn._TreeEnsembleClassifier(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Y_output", "Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['treeensembleclassifier'] = TreeEnsembleClassifier





class ZipMap:
    name = None
    classlabels_strings = None
    X_input = None
    Z_output = None

    #parameters
    classlabels_int64s = None

    def __init__(self, name):
        self.name = name
        self.Module = nn._ZipMap(name)
    def input(self, *args):
        inpts = ["X_input"]
        for i, x in enumerate(args):
            self.__dict__[inpts[i]] = x

    def output(self, *args):
        outputs = ["Z_output"]
        for i, x in enumerate(args):
            self.__dict__[outputs[i]] = x


    def call(self):
        pass

layer_map['zipmap'] = ZipMap

