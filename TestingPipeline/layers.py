layer_map = {}
#from _backend.nn import LSTM as c_LSTM
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




#from _backend.nn import Identity as c_Identity
class Identity:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Abs as c_Abs
class Abs:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import BatchNormalization as c_BatchNormalization
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




#from _backend.nn import Mean as c_Mean
class Mean:
    name = None
    mean_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Add as c_Add
class Add:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import GlobalMaxPool as c_GlobalMaxPool
class GlobalMaxPool:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Cast as c_Cast
class Cast:
    name = None
    input_input = None
    output_output = None

    #parameters
    to = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import AveragePool as c_AveragePool
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




#from _backend.nn import And as c_And
class And:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LRN as c_LRN
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




#from _backend.nn import ArgMax as c_ArgMax
class ArgMax:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axis = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Resize as c_Resize
class Resize:
    name = None
    X_input = None
    scales_input = None
    Y_output = None

    #parameters
    mode = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Expand as c_Expand
class Expand:
    name = None
    input_input = None
    shape_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Neg as c_Neg
class Neg:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Mul as c_Mul
class Mul:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ArgMin as c_ArgMin
class ArgMin:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axis = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import CastMap as c_CastMap
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




#from _backend.nn import Exp as c_Exp
class Exp:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Div as c_Div
class Div:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReverseSequence as c_ReverseSequence
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




#from _backend.nn import Ceil as c_Ceil
class Ceil:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import DepthToSpace as c_DepthToSpace
class DepthToSpace:
    name = None
    input_input = None
    output_output = None

    #parameters
    blocksize = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Clip as c_Clip
class Clip:
    name = None
    input_input = None
    output_output = None

    #parameters
    max = None
    min = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import RNN as c_RNN
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




#from _backend.nn import Concat as c_Concat
class Concat:
    name = None
    concat_result_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Constant as c_Constant
class Constant:
    name = None
    value = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LpPool as c_LpPool
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




#from _backend.nn import Conv as c_Conv
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




#from _backend.nn import Not as c_Not
class Not:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Gather as c_Gather
class Gather:
    name = None
    data_input = None
    indices_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ConvTranspose as c_ConvTranspose
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




#from _backend.nn import Dropout as c_Dropout
class Dropout:
    name = None
    data_input = None
    output_output = None
    mask_output_opt = None

    #parameters
    ratio = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LeakyRelu as c_LeakyRelu
class LeakyRelu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Elu as c_Elu
class Elu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import GlobalAveragePool as c_GlobalAveragePool
class GlobalAveragePool:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Gemm as c_Gemm
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




#from _backend.nn import MaxPool as c_MaxPool
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




#from _backend.nn import Equal as c_Equal
class Equal:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Tile as c_Tile
class Tile:
    name = None
    input_input = None
    repeats_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Flatten as c_Flatten
class Flatten:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Floor as c_Floor
class Floor:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import GRU as c_GRU
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




#from _backend.nn import GlobalLpPool as c_GlobalLpPool
class GlobalLpPool:
    name = None
    X_input = None
    Y_output = None

    #parameters
    p = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Greater as c_Greater
class Greater:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import HardSigmoid as c_HardSigmoid
class HardSigmoid:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None
    beta = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Selu as c_Selu
class Selu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None
    gamma = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Hardmax as c_Hardmax
class Hardmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import If as c_If
class If:
    name = None
    cond_input = None

    #parameters
    else_branch = None
    then_branch = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Min as c_Min
class Min:
    name = None
    min_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import InstanceNormalization as c_InstanceNormalization
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




#from _backend.nn import Less as c_Less
class Less:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import EyeLike as c_EyeLike
class EyeLike:
    name = None
    input_input = None
    output_output = None

    #parameters
    dtype = None
    k = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import RandomNormal as c_RandomNormal
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




#from _backend.nn import Slice as c_Slice
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




#from _backend.nn import PRelu as c_PRelu
class PRelu:
    name = None
    X_input = None
    slope_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Log as c_Log
class Log:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LogSoftmax as c_LogSoftmax
class LogSoftmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Loop as c_Loop
class Loop:
    name = None
    M_input_opt = None
    cond_input_opt = None

    #parameters
    body = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LpNormalization as c_LpNormalization
class LpNormalization:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None
    p = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import MatMul as c_MatMul
class MatMul:
    name = None
    A_input = None
    B_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceL2 as c_ReduceL2
class ReduceL2:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Max as c_Max
class Max:
    name = None
    max_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import MaxRoiPool as c_MaxRoiPool
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




#from _backend.nn import Or as c_Or
class Or:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Pad as c_Pad
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




#from _backend.nn import RandomUniformLike as c_RandomUniformLike
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




#from _backend.nn import Reciprocal as c_Reciprocal
class Reciprocal:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Pow as c_Pow
class Pow:
    name = None
    X_input = None
    Y_input = None
    Z_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import RandomNormalLike as c_RandomNormalLike
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




#from _backend.nn import OneHot as c_OneHot
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




#from _backend.nn import RandomUniform as c_RandomUniform
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




#from _backend.nn import ReduceL1 as c_ReduceL1
class ReduceL1:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceLogSum as c_ReduceLogSum
class ReduceLogSum:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceLogSumExp as c_ReduceLogSumExp
class ReduceLogSumExp:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceMax as c_ReduceMax
class ReduceMax:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import OneHotEncoder as c_OneHotEncoder
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




#from _backend.nn import IsNaN as c_IsNaN
class IsNaN:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceMean as c_ReduceMean
class ReduceMean:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceMin as c_ReduceMin
class ReduceMin:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import TreeEnsembleRegressor as c_TreeEnsembleRegressor
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




#from _backend.nn import ReduceProd as c_ReduceProd
class ReduceProd:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceSum as c_ReduceSum
class ReduceSum:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ReduceSumSquare as c_ReduceSumSquare
class ReduceSumSquare:
    name = None
    data_input = None
    reduced_output = None

    #parameters
    axes = None
    keepdims = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Relu as c_Relu
class Relu:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Reshape as c_Reshape
class Reshape:
    name = None
    data_input = None
    shape_input = None
    reshaped_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Shape as c_Shape
class Shape:
    name = None
    data_input = None
    shape_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Sigmoid as c_Sigmoid
class Sigmoid:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Size as c_Size
class Size:
    name = None
    data_input = None
    size_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Softmax as c_Softmax
class Softmax:
    name = None
    input_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Softplus as c_Softplus
class Softplus:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Softsign as c_Softsign
class Softsign:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import SpaceToDepth as c_SpaceToDepth
class SpaceToDepth:
    name = None
    input_input = None
    output_output = None

    #parameters
    blocksize = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import TfIdfVectorizer as c_TfIdfVectorizer
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




#from _backend.nn import Split as c_Split
class Split:
    name = None
    input_input = None

    #parameters
    axis = None
    split = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Imputer as c_Imputer
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




#from _backend.nn import Sqrt as c_Sqrt
class Sqrt:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Squeeze as c_Squeeze
class Squeeze:
    name = None
    data_input = None
    squeezed_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import TopK as c_TopK
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




#from _backend.nn import Sub as c_Sub
class Sub:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Sum as c_Sum
class Sum:
    name = None
    sum_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Shrink as c_Shrink
class Shrink:
    name = None
    input_input = None
    output_output = None

    #parameters
    bias = None
    lambd = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Tanh as c_Tanh
class Tanh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Transpose as c_Transpose
class Transpose:
    name = None
    data_input = None
    transposed_output = None

    #parameters
    perm = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Unsqueeze as c_Unsqueeze
class Unsqueeze:
    name = None
    data_input = None
    expanded_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Upsample as c_Upsample
class Upsample:
    name = None
    X_input = None
    scales_input = None
    Y_output = None

    #parameters
    mode = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import SVMClassifier as c_SVMClassifier
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




#from _backend.nn import Xor as c_Xor
class Xor:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Acos as c_Acos
class Acos:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Asin as c_Asin
class Asin:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Atan as c_Atan
class Atan:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Cos as c_Cos
class Cos:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Sin as c_Sin
class Sin:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Tan as c_Tan
class Tan:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Multinomial as c_Multinomial
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




#from _backend.nn import Scan as c_Scan
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




#from _backend.nn import Compress as c_Compress
class Compress:
    name = None
    input_input = None
    condition_input = None
    output_output = None

    #parameters
    axis = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ConstantOfShape as c_ConstantOfShape
class ConstantOfShape:
    name = None
    value = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import MaxUnpool as c_MaxUnpool
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




#from _backend.nn import Scatter as c_Scatter
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




#from _backend.nn import Sinh as c_Sinh
class Sinh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Cosh as c_Cosh
class Cosh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Asinh as c_Asinh
class Asinh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Acosh as c_Acosh
class Acosh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import NonMaxSuppression as c_NonMaxSuppression
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




#from _backend.nn import Atanh as c_Atanh
class Atanh:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Sign as c_Sign
class Sign:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Erf as c_Erf
class Erf:
    name = None
    input_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Where as c_Where
class Where:
    name = None
    condition_input = None
    X_input = None
    Y_input = None
    output_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import NonZero as c_NonZero
class NonZero:
    name = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import MeanVarianceNormalization as c_MeanVarianceNormalization
class MeanVarianceNormalization:
    name = None
    X_input = None
    Y_output = None

    #parameters
    axes = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import StringNormalizer as c_StringNormalizer
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




#from _backend.nn import Mod as c_Mod
class Mod:
    name = None
    A_input = None
    B_input = None
    C_output = None

    #parameters
    fmod = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import ThresholdedRelu as c_ThresholdedRelu
class ThresholdedRelu:
    name = None
    X_input = None
    Y_output = None

    #parameters
    alpha = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import MatMulInteger as c_MatMulInteger
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




#from _backend.nn import QLinearMatMul as c_QLinearMatMul
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




#from _backend.nn import ConvInteger as c_ConvInteger
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




#from _backend.nn import QLinearConv as c_QLinearConv
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




#from _backend.nn import QuantizeLinear as c_QuantizeLinear
class QuantizeLinear:
    name = None
    x_input = None
    y_scale_input = None
    y_zero_point_input_opt = None
    y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import DequantizeLinear as c_DequantizeLinear
class DequantizeLinear:
    name = None
    x_input = None
    x_scale_input = None
    x_zero_point_input_opt = None
    y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import IsInf as c_IsInf
class IsInf:
    name = None
    X_input = None
    Y_output = None

    #parameters
    detect_negative = None
    detect_positive = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import RoiAlign as c_RoiAlign
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




#from _backend.nn import ArrayFeatureExtractor as c_ArrayFeatureExtractor
class ArrayFeatureExtractor:
    name = None
    X_input = None
    Y_input = None
    Z_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import Binarizer as c_Binarizer
class Binarizer:
    name = None
    X_input = None
    Y_output = None

    #parameters
    threshold = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import CategoryMapper as c_CategoryMapper
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




#from _backend.nn import DictVectorizer as c_DictVectorizer
class DictVectorizer:
    name = None
    string_vocabulary = None
    X_input = None
    Y_output = None

    #parameters
    int64_vocabulary = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import FeatureVectorizer as c_FeatureVectorizer
class FeatureVectorizer:
    name = None
    Y_output = None

    #parameters
    inputdimensions = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import LabelEncoder as c_LabelEncoder
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




#from _backend.nn import LinearClassifier as c_LinearClassifier
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




#from _backend.nn import LinearRegressor as c_LinearRegressor
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




#from _backend.nn import Normalizer as c_Normalizer
class Normalizer:
    name = None
    X_input = None
    Y_output = None

    #parameters
    norm = None

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import SVMRegressor as c_SVMRegressor
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




#from _backend.nn import Scaler as c_Scaler
class Scaler:
    name = None
    offset = None
    scale = None
    X_input = None
    Y_output = None

    #parameters

    def __init__(self, name):
        self.name = name

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




#from _backend.nn import TreeEnsembleClassifier as c_TreeEnsembleClassifier
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




#from _backend.nn import ZipMap as c_ZipMap
class ZipMap:
    name = None
    classlabels_strings = None
    X_input = None
    Z_output = None

    #parameters
    classlabels_int64s = None

    def __init__(self, name):
        self.name = name

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

