import numpy as np
import _backend.nn as nn
import onnx.helper
from onnx.backend.test.case.node import expect
layer_map = {}
tensors = {}


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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LSTM
        self.run_ = nn._LSTM_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.input_forget, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.initial_c_i, self.P_i, self.Y_o, self.Y_h_o, self.Y_c_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEFAULTS():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 3
            weight_scale = 0.1
            number_of_gates = 4
            
            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            
            lstm = LSTM_Helper(X=input, W=W, R=R)
            _, Y_h = lstm.step()
            expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')
        def _INITIAL_BIAS():
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)
            
            input_size = 3
            hidden_size = 4
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 4
            
            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            
            # Adding custom bias
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), 1)
            
            lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
            _, Y_h = lstm.step()
            expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_with_initial_bias')
        def _PEEPHOLES():
            input = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(np.float32)
            
            input_size = 4
            hidden_size = 3
            weight_scale = 0.1
            number_of_gates = 4
            number_of_peepholes = 3
            
            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            # Initializing Inputs
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
            seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
            init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
            init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
            P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32)
            
            lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
            _, Y_h = lstm.step()
            expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[Y_h.astype(np.float32)],
                   name='test_lstm_with_peepholes')
        _DEFAULTS()
        _INITIAL_BIAS()
        _PEEPHOLES()

layer_map['LSTM'] = LSTM





class Identity:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Identity
        self.run_ = nn._Identity_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _IDENTITY():
            node = onnx.helper.make_node(
                'Identity',
                inputs=['x'],
                outputs=['y'],
            )
            
            data = np.array([[[
                [1, 2],
                [3, 4],
            ]]], dtype=np.float32)
            
            expect(node, inputs=[data], outputs=[data],
                   name='test_identity')
        _IDENTITY()

layer_map['Identity'] = Identity





class Abs:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Abs
        self.run_ = nn._Abs_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ABS():
            node = onnx.helper.make_node(
                'Abs',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = abs(x)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_abs')
        _ABS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._BatchNormalization
        self.run_ = nn._BatchNormalization_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.epsilon, self.momentum, self.X_i, self.scale_i, self.B_i, self.mean_i, self.var_i, self.Y_o, self.mean_o, self.var_o, self.saved_mean_o, self.saved_var_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _BATCHNORMALIZATION():
            def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
                dims_x = len(x.shape)
                dim_ones = (1,) * (dims_x - 2)
                s = s.reshape(-1, *dim_ones)
                bias = bias.reshape(-1, *dim_ones)
                mean = mean.reshape(-1, *dim_ones)
                var = var.reshape(-1, *dim_ones)
                return s * (x - mean) / np.sqrt(var + epsilon) + bias
            
            # input size: (1, 2, 1, 3)
            x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
            s = np.array([1.0, 1.5]).astype(np.float32)
            bias = np.array([0, 1]).astype(np.float32)
            mean = np.array([0, 3]).astype(np.float32)
            var = np.array([1, 1.5]).astype(np.float32)
            y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)
            
            node = onnx.helper.make_node(
                'BatchNormalization',
                inputs=['x', 's', 'bias', 'mean', 'var'],
                outputs=['y'],
            )
            
            # output size: (1, 2, 1, 3)
            expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
                   name='test_batchnorm_example')
            
            # input size: (2, 3, 4, 5)
            x = np.random.randn(2, 3, 4, 5).astype(np.float32)
            s = np.random.randn(3).astype(np.float32)
            bias = np.random.randn(3).astype(np.float32)
            mean = np.random.randn(3).astype(np.float32)
            var = np.random.rand(3).astype(np.float32)
            epsilon = 1e-2
            y = _batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)
            
            node = onnx.helper.make_node(
                'BatchNormalization',
                inputs=['x', 's', 'bias', 'mean', 'var'],
                outputs=['y'],
                epsilon=epsilon,
            )
            
            # output size: (2, 3, 4, 5)
            expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
                   name='test_batchnorm_epsilon')
        _BATCHNORMALIZATION()

layer_map['BatchNormalization'] = BatchNormalization





class Mean:
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
    mean_o = str()

    #parameters

    input_params = ["x0_i", "x1_i", "x2_i", "x3_i", "x4_i", "x5_i", "x6_i", "x7_i", "x8_i", "x9_i", "x10_i", "x11_i", "x12_i", "x13_i", "x14_i", "x15_i", "x16_i", "x17_i", "x18_i", "x19_i", "x20_i", "x21_i", "x22_i", "x23_i", "x24_i", "x25_i", "x26_i", "x27_i", "x28_i", "x29_i", "x30_i", "x31_i"]
    output_params = ["mean_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Mean
        self.run_ = nn._Mean_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.mean_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MEAN():
            data_0 = np.array([3, 0, 2]).astype(np.float32)
            data_1 = np.array([1, 3, 4]).astype(np.float32)
            data_2 = np.array([2, 6, 6]).astype(np.float32)
            result = np.array([2, 3, 4]).astype(np.float32)
            node = onnx.helper.make_node(
                'Mean',
                inputs=['data_0', 'data_1', 'data_2'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
                   name='test_mean_example')
            
            node = onnx.helper.make_node(
                'Mean',
                inputs=['data_0'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0], outputs=[data_0],
                   name='test_mean_one_input')
            
            result = np.divide(np.add(data_0, data_1), 2.)
            node = onnx.helper.make_node(
                'Mean',
                inputs=['data_0', 'data_1'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1], outputs=[result],
                   name='test_mean_two_inputs')
        _MEAN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Add
        self.run_ = nn._Add_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ADD():
            node = onnx.helper.make_node(
                'Add',
                inputs=['x', 'y'],
                outputs=['sum'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            expect(node, inputs=[x, y], outputs=[x + y],
                   name='test_add')
        def _ADD_BROADCAST():
            node = onnx.helper.make_node(
                'Add',
                inputs=['x', 'y'],
                outputs=['sum'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(5).astype(np.float32)
            expect(node, inputs=[x, y], outputs=[x + y],
                   name='test_add_bcast')
        _ADD()
        _ADD_BROADCAST()

layer_map['Add'] = Add





class GlobalMaxPool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._GlobalMaxPool
        self.run_ = nn._GlobalMaxPool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _GLOBALMAXPOOL():
            
            node = onnx.helper.make_node(
                'GlobalMaxPool',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(1, 3, 5, 5).astype(np.float32)
            spatial_shape = np.ndim(x) - 2
            y = np.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
            for _ in range(spatial_shape):
                y = np.expand_dims(y, -1)
            expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool')
        def _GLOBALMAXPOOL_PRECOMPUTED():
            
            node = onnx.helper.make_node(
                'GlobalMaxPool',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.array([[[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]]]).astype(np.float32)
            y = np.array([[[[9]]]]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool_precomputed')
        _GLOBALMAXPOOL()
        _GLOBALMAXPOOL_PRECOMPUTED()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Cast
        self.run_ = nn._Cast_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.to, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CAST():
            shape = (3, 4)
            test_cases = [
                ('FLOAT', 'FLOAT16'),
                ('FLOAT', 'DOUBLE'),
                ('FLOAT16', 'FLOAT'),
                ('FLOAT16', 'DOUBLE'),
                ('DOUBLE', 'FLOAT'),
                ('DOUBLE', 'FLOAT16'),
                ('FLOAT', 'STRING'),
                ('STRING', 'FLOAT'),
            ]
            
            for from_type, to_type in test_cases:
                if 'STRING' != from_type:
                    input = np.random.random_sample(shape).astype(
                        TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
                    if ('STRING' == to_type):
                        # Converting input to str, then give it np.object dtype for generating script
                        ss = []
                        for i in input.flatten():
                            s = str(i).encode('utf-8')
                            su = s.decode('utf-8')
                            ss.append(su)
            
                        output = np.array(ss).astype(np.object).reshape([3, 4])
                    else:
                        output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
                else:
                    input = np.array([u'0.47892547', u'0.48033667', u'0.49968487', u'0.81910545',
                        u'0.47031248', u'0.816468', u'0.21087195', u'0.7229038',
                        u'NaN', u'INF', u'+INF', u'-INF'], dtype=np.dtype(np.object)).reshape([3, 4])
                    output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
                node = onnx.helper.make_node(
                    'Cast',
                    inputs=['input'],
                    outputs=['output'],
                    to=getattr(TensorProto, to_type),
                )
                expect(node, inputs=[input], outputs=[output],
                           name='test_cast_' + from_type + '_to_' + to_type)
        _CAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._AveragePool
        self.run_ = nn._AveragePool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.ceil_mode, self.count_include_pad, self.pads, self.strides, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _AVERAGEPOOL_2D_PRECOMPUTED_PADS():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 5, 5]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2]
            
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[7, 7.5, 8, 8.5, 9],
                            [9.5, 10, 10.5, 11, 11.5],
                            [12, 12.5, 13, 13.5, 14],
                            [14.5, 15, 15.5, 16, 16.5],
                            [17, 17.5, 18, 18.5, 19]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_pads')
        def _AVERAGEPOOL_2D_PRECOMPUTED_PADS_COUNT_INCLUDE_PAD():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 5, 5]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2],
                count_include_pad=1
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                            [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                            [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                            [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                            [6.1200, 8.4000, 10.8000, 8.8800, 6.8400]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_pads_count_include_pad')
        def _AVERAGEPOOL_2D_PRECOMPUTED_STRIDES():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[4, 6],
                            [14, 16]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_strides')
        def _AVERAGEPOOL_2D_PRECOMPUTED_SAME_UPPER():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 3, 3]
            pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[4, 5.5, 7],
                            [11.5, 13, 14.5],
                            [19, 20.5, 22]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_same_upper')
        def _AVERAGEPOOL_1D_DEFAULT():
            """
            input_shape: [1, 3, 32]
            output_shape: [1, 3, 31]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2],
            )
            x = np.random.randn(1, 3, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2]
            strides = [1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_1d_default')
        def _AVERAGEPOOL_2D_DEFAULT():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 31, 31]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_default')
        def _AVERAGEPOOL_3D_DEFAULT():
            """
            input_shape: [1, 3, 32, 32, 32]
            output_shape: [1, 3, 31, 31, 31]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2, 2],
            )
            x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2, 2, 2]
            strides = [1, 1, 1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_3d_default')
        def _AVERAGEPOOL_2D_SAME_UPPER():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 32, 32]
            pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_top = pad_shape[0] // 2
            pad_bottom = pad_shape[0] - pad_top
            pad_left = pad_shape[1] // 2
            pad_right = pad_shape[1] - pad_left
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_upper')
        def _AVERAGEPOOL_2D_SAME_LOWER():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 32, 32]
            pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_LOWER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_bottom = pad_shape[0] // 2
            pad_top = pad_shape[0] - pad_bottom
            pad_right = pad_shape[1] // 2
            pad_left = pad_shape[1] - pad_right
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_lower')
        def _AVERAGEPOOL_2D_PADS():
            """
            input_shape: [1, 3, 28, 28]
            output_shape: [1, 3, 30, 30]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[2, 2, 2, 2]
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (1, 1)
            pad_bottom = 2
            pad_top = 2
            pad_right = 2
            pad_left = 2
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads')
        def _AVERAGEPOOL_2D_PADS_COUNT_INCLUDE_PAD():
            """
            input_shape: [1, 3, 28, 28]
            output_shape: [1, 3, 30, 30]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[2, 2, 2, 2],
                count_include_pad=1,
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (1, 1)
            pad_bottom = 2
            pad_top = 2
            pad_right = 2
            pad_left = 2
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=0)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG', count_include_pad=1)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads_count_include_pad')
        def _AVERAGEPOOL_2D_STRIDES():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 10, 10]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                strides=[3, 3]
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (5, 5)
            strides = (3, 3)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_strides')
        def _AVERAGEPOOL_2D_CEIL():
            """
            input_shape: [1, 1, 4, 4]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'AveragePool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=True
            )
            x = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]]]).astype(np.float32)
            y = np.array([[[
                [6, 7.5],
                [12, 13.5]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_ceil')
        _AVERAGEPOOL_2D_PRECOMPUTED_PADS()
        _AVERAGEPOOL_2D_PRECOMPUTED_PADS_COUNT_INCLUDE_PAD()
        _AVERAGEPOOL_2D_PRECOMPUTED_STRIDES()
        _AVERAGEPOOL_2D_PRECOMPUTED_SAME_UPPER()
        _AVERAGEPOOL_1D_DEFAULT()
        _AVERAGEPOOL_2D_DEFAULT()
        _AVERAGEPOOL_3D_DEFAULT()
        _AVERAGEPOOL_2D_SAME_UPPER()
        _AVERAGEPOOL_2D_SAME_LOWER()
        _AVERAGEPOOL_2D_PADS()
        _AVERAGEPOOL_2D_PADS_COUNT_INCLUDE_PAD()
        _AVERAGEPOOL_2D_STRIDES()
        _AVERAGEPOOL_2D_CEIL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._And
        self.run_ = nn._And_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _AND():
            node = onnx.helper.make_node(
                'And',
                inputs=['x', 'y'],
                outputs=['and'],
            )
            
            # 2d
            x = (np.random.randn(3, 4) > 0).astype(np.bool)
            y = (np.random.randn(3, 4) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and2d')
            
            # 3d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and3d')
            
            # 4d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and4d')
        def _AND_BROADCAST():
            node = onnx.helper.make_node(
                'And',
                inputs=['x', 'y'],
                outputs=['and'],
            )
            
            # 3d vs 1d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(5) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and_bcast3v1d')
            
            # 3d vs 2d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(4, 5) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and_bcast3v2d')
            
            # 4d vs 2d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(5, 6) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and_bcast4v2d')
            
            # 4d vs 3d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and_bcast4v3d')
            
            # 4d vs 4d
            x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
            z = np.logical_and(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_and_bcast4v4d')
        _AND()
        _AND_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LRN
        self.run_ = nn._LRN_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.size, self.alpha, self.beta, self.bias, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _LRN():
            alpha = 0.0002
            beta = 0.5
            bias = 2.0
            nsize = 3
            node = onnx.helper.make_node(
                'LRN',
                inputs=['x'],
                outputs=['y'],
                alpha=alpha,
                beta=beta,
                bias=bias,
                size=nsize
            )
            x = np.random.randn(5, 5, 5, 5).astype(np.float32)
            square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
            for n, c, h, w in np.ndindex(x.shape):
                square_sum[n, c, h, w] = sum(x[n,
                                               max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                               h,
                                               w] ** 2)
            y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
            expect(node, inputs=[x], outputs=[y],
                   name='test_lrn')
        def _DEFAULT():
            alpha = 0.0001
            beta = 0.75
            bias = 1.0
            nsize = 3
            node = onnx.helper.make_node(
                'LRN',
                inputs=['x'],
                outputs=['y'],
                size=3
            )
            x = np.random.randn(5, 5, 5, 5).astype(np.float32)
            square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
            for n, c, h, w in np.ndindex(x.shape):
                square_sum[n, c, h, w] = sum(x[n,
                                               max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                               h,
                                               w] ** 2)
            y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
            expect(node, inputs=[x], outputs=[y],
                   name='test_lrn_default')
        _LRN()
        _DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ArgMax
        self.run_ = nn._ArgMax_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NO_KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            axis = 1
            keepdims = 0
            node = onnx.helper.make_node(
                'ArgMax',
                inputs=['data'],
                outputs=['result'],
                axis=axis,
                keepdims=keepdims)
            # result: [[0, 1]]
            result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [2, 4]
            result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_random')
        def _KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            axis = 1
            keepdims = 1
            node = onnx.helper.make_node(
                'ArgMax',
                inputs=['data'],
                outputs=['result'],
                axis=axis,
                keepdims=keepdims)
            # result: [[0], [1]]
            result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [2, 1, 4]
            result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            keepdims = 1
            node = onnx.helper.make_node(
                'ArgMax',
                inputs=['data'],
                outputs=['result'],
                keepdims=keepdims)
            
            # result: [[1], [1]]
            result = argmax_use_numpy(data, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [1, 3, 4]
            result = argmax_use_numpy(data, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_random')
        _NO_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Resize
        self.run_ = nn._Resize_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.mode, self.X_i, self.scales_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _UPSAMPLE_NEAREST():
            node = onnx.helper.make_node(
                'Resize',
                inputs=['X', 'scales'],
                outputs=['Y'],
                mode='nearest',
            )
            
            data = np.array([[[
                [1, 2],
                [3, 4],
            ]]], dtype=np.float32)
            
            scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
            
            output = np.array([[[
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4],
                [3, 3, 3, 4, 4, 4],
            ]]], dtype=np.float32)
            
            expect(node, inputs=[data, scales], outputs=[output],
                   name='test_resize_upsample_nearest')
        def _DOWNSAMPLE_NEAREST():
            node = onnx.helper.make_node(
                'Resize',
                inputs=['X', 'scales'],
                outputs=['Y'],
                mode='nearest',
            )
            
            data = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]]], dtype=np.float32)
            
            scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
            
            output = np.array([[[
                [1, 3]
            ]]], dtype=np.float32)
            
            expect(node, inputs=[data, scales], outputs=[output],
                   name='test_resize_downsample_nearest')
        def _UPSAMPLE_LINEAR():
            node = onnx.helper.make_node(
                'Resize',
                inputs=['X', 'scales'],
                outputs=['Y'],
                mode='linear',
            )
            
            data = np.array([[[
                [1, 2],
                [3, 4],
            ]]], dtype=np.float32)
            
            scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
            
            output = np.array([[[
                [1, 1.5, 2, 2],
                [2, 2.5, 3, 3],
                [3, 3.5, 4, 4],
                [3, 3.5, 4, 4],
            ]]], dtype=np.float32)
            
            expect(node, inputs=[data, scales], outputs=[output],
                   name='test_resize_upsample_linear')
        def _DOWNSAMPLE_LINEAR():
            node = onnx.helper.make_node(
                'Resize',
                inputs=['X', 'scales'],
                outputs=['Y'],
                mode='linear',
            )
            
            data = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]]], dtype=np.float32)
            
            scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
            
            output = np.array([[[
                [1, 2.66666651]
            ]]], dtype=np.float32)
            
            expect(node, inputs=[data, scales], outputs=[output],
                   name='test_resize_downsample_linear')
        _UPSAMPLE_NEAREST()
        _DOWNSAMPLE_NEAREST()
        _UPSAMPLE_LINEAR()
        _DOWNSAMPLE_LINEAR()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Expand
        self.run_ = nn._Expand_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.shape_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DIM_CHANGED():
            node = onnx.helper.make_node(
                'Expand',
                inputs=['data', 'new_shape'],
                outputs=['expanded'],
            )
            shape = [3, 1]
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[1.], [2.], [3.]]
            new_shape = [2, 1, 6]
            expanded = data * np.ones(new_shape, dtype=np.float32)
            #print(expanded)
            #[[[1., 1., 1., 1., 1., 1.],
            #  [2., 2., 2., 2., 2., 2.],
            #  [3., 3., 3., 3., 3., 3.]],
            #
            # [[1., 1., 1., 1., 1., 1.],
            #  [2., 2., 2., 2., 2., 2.],
            #  [3., 3., 3., 3., 3., 3.]]]
            new_shape = np.array(new_shape, dtype=np.int64)
            expect(node, inputs=[data, new_shape], outputs=[expanded],
                   name='test_expand_dim_changed')
        def _DIM_UNCHANGED():
            node = onnx.helper.make_node(
                'Expand',
                inputs=['data', 'new_shape'],
                outputs=['expanded'],
            )
            shape = [3, 1]
            new_shape = [3, 4]
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[1.], [2.], [3.]]
            expanded = np.tile(data, 4)
            #print(expanded)
            #[[1., 1., 1., 1.],
            # [2., 2., 2., 2.],
            # [3., 3., 3., 3.]]
            new_shape = np.array(new_shape, dtype=np.int64)
            expect(node, inputs=[data, new_shape], outputs=[expanded],
                   name='test_expand_dim_unchanged')
        _DIM_CHANGED()
        _DIM_UNCHANGED()

layer_map['Expand'] = Expand





class Neg:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Neg
        self.run_ = nn._Neg_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NEG():
            node = onnx.helper.make_node(
                'Neg',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-4, 2]).astype(np.float32)
            y = np.negative(x)  # expected output [4., -2.],
            expect(node, inputs=[x], outputs=[y],
                   name='test_neg_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.negative(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_neg')
        _NEG()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Mul
        self.run_ = nn._Mul_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MUL():
            node = onnx.helper.make_node(
                'Mul',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([1, 2, 3]).astype(np.float32)
            y = np.array([4, 5, 6]).astype(np.float32)
            z = x * y  # expected output [4., 10., 18.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mul_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            z = x * y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mul')
        def _MUL_BROADCAST():
            node = onnx.helper.make_node(
                'Mul',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(5).astype(np.float32)
            z = x * y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mul_bcast')
        _MUL()
        _MUL_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ArgMin
        self.run_ = nn._ArgMin_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NO_KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            axis = 1
            keepdims = 0
            node = onnx.helper.make_node(
                'ArgMin',
                inputs=['data'],
                outputs=['result'],
                axis=axis,
                keepdims=keepdims)
            # result: [[1, 0]]
            result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [2, 4]
            result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_random')
        def _KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            axis = 1
            keepdims = 1
            node = onnx.helper.make_node(
                'ArgMin',
                inputs=['data'],
                outputs=['result'],
                axis=axis,
                keepdims=keepdims)
            # result: [[1], [0]]
            result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [2, 1, 4]
            result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            data = np.array([[2, 1], [3, 10]], dtype=np.float32)
            keepdims = 1
            node = onnx.helper.make_node(
                'ArgMin',
                inputs=['data'],
                outputs=['result'],
                keepdims=keepdims)
            
            # result: [[0], [0]]
            result = argmin_use_numpy(data, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_example')
            
            data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
            # result's shape: [1, 3, 4]
            result = argmin_use_numpy(data, keepdims=keepdims)
            expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_random')
        _NO_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._CastMap
        self.run_ = nn._CastMap_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.cast_to, self.map_form, self.max_map, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['CastMap'] = CastMap





class Exp:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Exp
        self.run_ = nn._Exp_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _EXP():
            node = onnx.helper.make_node(
                'Exp',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.exp(x)  # expected output [0.36787945, 1., 2.71828175]
            expect(node, inputs=[x], outputs=[y],
                   name='test_exp_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.exp(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_exp')
        _EXP()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Div
        self.run_ = nn._Div_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DIV():
            node = onnx.helper.make_node(
                'Div',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([3, 4]).astype(np.float32)
            y = np.array([1, 2]).astype(np.float32)
            z = x / y  # expected output [3., 2.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_div_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
            z = x / y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_div')
        def _DIV_BROADCAST():
            node = onnx.helper.make_node(
                'Div',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.rand(5).astype(np.float32) + 1.0
            z = x / y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_div_bcast')
        _DIV()
        _DIV_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReverseSequence
        self.run_ = nn._ReverseSequence_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.batch_axis, self.time_axis, self.input_i, self.sequence_lens_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _REVERSESEQUENCE_TIME():
            node = onnx.helper.make_node(
                'ReverseSequence',
                inputs=['x', 'sequence_lens'],
                outputs=['y'],
                time_axis=0,
                batch_axis=1,
            )
            x = np.array([[0.0, 4.0, 8.0, 12.0],
                          [1.0, 5.0, 9.0, 13.0],
                          [2.0, 6.0, 10.0, 14.0],
                          [3.0, 7.0, 11.0, 15.0]], dtype=np.float32)
            sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
            
            y = np.array([[3.0, 6.0, 9.0, 12.0],
                          [2.0, 5.0, 8.0, 13.0],
                          [1.0, 4.0, 10.0, 14.0],
                          [0.0, 7.0, 11.0, 15.0]], dtype=np.float32)
            
            expect(node, inputs=[x, sequence_lens], outputs=[y],
                   name='test_reversesequence_time')
        def _REVERSESEQUENCE_BATCH():
            node = onnx.helper.make_node(
                'ReverseSequence',
                inputs=['x', 'sequence_lens'],
                outputs=['y'],
                time_axis=1,
                batch_axis=0,
            )
            x = np.array([[0.0, 1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0, 7.0],
                          [8.0, 9.0, 10.0, 11.0],
                          [12.0, 13.0, 14.0, 15.0]], dtype=np.float32)
            sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)
            
            y = np.array([[0.0, 1.0, 2.0, 3.0],
                          [5.0, 4.0, 6.0, 7.0],
                          [10.0, 9.0, 8.0, 11.0],
                          [15.0, 14.0, 13.0, 12.0]], dtype=np.float32)
            
            expect(node, inputs=[x, sequence_lens], outputs=[y],
                   name='test_reversesequence_batch')
        _REVERSESEQUENCE_TIME()
        _REVERSESEQUENCE_BATCH()

layer_map['ReverseSequence'] = ReverseSequence





class Ceil:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Ceil
        self.run_ = nn._Ceil_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CEIL():
            node = onnx.helper.make_node(
                'Ceil',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1.5, 1.2]).astype(np.float32)
            y = np.ceil(x)  # expected output [-1., 2.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_ceil_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.ceil(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_ceil')
        _CEIL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._DepthToSpace
        self.run_ = nn._DepthToSpace_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.blocksize, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEPTHTOSPACE():
            b, c, h, w = shape = (2, 8, 3, 3)
            blocksize = 2
            node = onnx.helper.make_node(
                'DepthToSpace',
                inputs=['x'],
                outputs=['y'],
                blocksize=blocksize,
            )
            x = np.random.random_sample(shape).astype(np.float32)
            tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
            tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
            y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
            expect(node, inputs=[x], outputs=[y],
                   name='test_depthtospace')
        def _EXAMPLE():
            node = onnx.helper.make_node(
                'DepthToSpace',
                inputs=['x'],
                outputs=['y'],
                blocksize=2,
            )
            
            # (1, 4, 2, 3) input tensor
            x = np.array([[[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]],
                           [[12, 13, 14],
                            [15, 16, 17]],
                           [[18, 19, 20],
                            [21, 22, 23]]]]).astype(np.float32)
            
            # (1, 1, 4, 6) output tensor
            y = np.array([[[[0, 6, 1, 7, 2, 8],
                            [12, 18, 13, 19, 14, 20],
                            [3, 9, 4, 10, 5, 11],
                            [15, 21, 16, 22, 17, 23]]]]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_depthtospace_example')
        _DEPTHTOSPACE()
        _EXAMPLE()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Clip
        self.run_ = nn._Clip_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.max, self.min, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CLIP():
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x'],
                outputs=['y'],
                min=-1.0,
                max=1.0
            )
            
            x = np.array([-2, 0, 2]).astype(np.float32)
            y = np.clip(x, -1, 1)  # expected output [-1., 0., 1.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, -1.0, 1.0)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip')
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x'],
                outputs=['y'],
                min=-5.0,
                max=5.0,
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.array([-1, 0, 1]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_inbounds')
            
            x = np.array([-6, 0, 6]).astype(np.float32)
            y = np.array([-5, 0, 5]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_outbounds')
            
            x = np.array([-1, 0, 6]).astype(np.float32)
            y = np.array([-1, 0, 5]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_splitbounds')
        def _CLIP_DEFAULT():
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x'],
                outputs=['y'],
                min=0.0
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0.0, np.inf)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_default_min')
            
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x'],
                outputs=['y'],
                max=0.0
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, -np.inf, 0.0)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_default_max')
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.array([-1, 0, 1]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_default_inbounds')
        _CLIP()
        _CLIP_DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RNN
        self.run_ = nn._RNN_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.Y_o, self.Y_h_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEFAULTS():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 4
            weight_scale = 0.1
            
            node = onnx.helper.make_node(
                'RNN',
                inputs=['X', 'W', 'R'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)
            
            rnn = RNN_Helper(X=input, W=W, R=R)
            _, Y_h = rnn.step()
            expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_simple_rnn_defaults')
        def _INITIAL_BIAS():
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)
            
            input_size = 3
            hidden_size = 5
            custom_bias = 0.1
            weight_scale = 0.1
            
            node = onnx.helper.make_node(
                'RNN',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)
            
            # Adding custom bias
            W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
            R_B = np.zeros((1, hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)
            
            rnn = RNN_Helper(X=input, W=W, R=R, B=B)
            _, Y_h = rnn.step()
            expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)],
                   name='test_simple_rnn_with_initial_bias')
        def _SEQ_LENGTH():
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                              [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)
            
            input_size = 3
            hidden_size = 5
            
            node = onnx.helper.make_node(
                'RNN',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = np.random.randn(1, hidden_size, input_size).astype(np.float32)
            R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32)
            
            # Adding custom bias
            W_B = np.random.randn(1, hidden_size).astype(np.float32)
            R_B = np.random.randn(1, hidden_size).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)
            
            rnn = RNN_Helper(X=input, W=W, R=R, B=B)
            _, Y_h = rnn.step()
            expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_rnn_seq_length')
        _DEFAULTS()
        _INITIAL_BIAS()
        _SEQ_LENGTH()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Concat
        self.run_ = nn._Concat_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.concat_result_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONCAT():
            test_cases = {
                '1d': ([1, 2],
                       [3, 4]),
                '2d': ([[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]),
                '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                       [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
            }  # type: Dict[Text, Sequence[Any]]
            
            for test_case, values_ in test_cases.items():
                values = [np.asarray(v, dtype=np.float32) for v in values_]
                for i in range(len(values[0].shape)):
                    in_args = ['value' + str(k) for k in range(len(values))]
                    node = onnx.helper.make_node(
                        'Concat',
                        inputs=[s for s in in_args],
                        outputs=['output'],
                        axis=i
                    )
                    output = np.concatenate(values, i)
                    expect(node, inputs=[v for v in values], outputs=[output],
                           name='test_concat_' + test_case + '_axis_' + str(i))
        _CONCAT()

layer_map['Concat'] = Concat





class Constant:
    name = None
    output_o = str()

    #parameters
    value = list()

    input_params = []
    output_params = ["output_o"]
    attribute_params = ["value"]
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Constant
        self.run_ = nn._Constant_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.value, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONSTANT():
            values = np.random.randn(5, 5).astype(np.float32)
            node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['values'],
                value=onnx.helper.make_tensor(
                    name='const_tensor',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=values.shape,
                    vals=values.flatten().astype(float),
                ),
            )
            
            expect(node, inputs=[], outputs=[values],
                   name='test_constant')
        _CONSTANT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LpPool
        self.run_ = nn._LpPool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.p, self.pads, self.strides, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Conv
        self.run_ = nn._Conv_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.X_i, self.W_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONV():
            
            x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.]]]]).astype(np.float32)
            W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                            [1., 1., 1.],
                            [1., 1., 1.]]]]).astype(np.float32)
            
            # Convolution with padding
            node_with_padding = onnx.helper.make_node(
                'Conv',
                inputs=['x', 'W'],
                outputs=['y'],
                kernel_shape=[3, 3],
                # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
                pads=[1, 1, 1, 1],
            )
            y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                         [33., 54., 63., 72., 51.],
                                         [63., 99., 108., 117., 81.],
                                         [93., 144., 153., 162., 111.],
                                         [72., 111., 117., 123., 84.]]]]).astype(np.float32)
            expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
                   name='test_basic_conv_with_padding')
            
            # Convolution without padding
            node_without_padding = onnx.helper.make_node(
                'Conv',
                inputs=['x', 'W'],
                outputs=['y'],
                kernel_shape=[3, 3],
                # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
                pads=[0, 0, 0, 0],
            )
            y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                            [99., 108., 117.],
                                            [144., 153., 162.]]]]).astype(np.float32)
            expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
                   name='test_basic_conv_without_padding')
        def _CONV_WITH_STRIDES():
            
            x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.],
                            [25., 26., 27., 28., 29.],
                            [30., 31., 32., 33., 34.]]]]).astype(np.float32)
            W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                            [1., 1., 1.],
                            [1., 1., 1.]]]]).astype(np.float32)
            
            # Convolution with strides=2 and padding
            node_with_padding = onnx.helper.make_node(
                'Conv',
                inputs=['x', 'W'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
            )
            y_with_padding = np.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                         [63., 108., 81.],
                                         [123., 198., 141.],
                                         [112., 177., 124.]]]]).astype(np.float32)
            expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
                   name='test_conv_with_strides_padding')
            
            # Convolution with strides=2 and no padding
            node_without_padding = onnx.helper.make_node(
                'Conv',
                inputs=['x', 'W'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[0, 0, 0, 0],
                strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
            )
            y_without_padding = np.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                            [144., 162.],
                                            [234., 252.]]]]).astype(np.float32)
            expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
                   name='test_conv_with_strides_no_padding')
            
            # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
            node_with_asymmetric_padding = onnx.helper.make_node(
                'Conv',
                inputs=['x', 'W'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[1, 0, 1, 0],
                strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
            )
            y_with_asymmetric_padding = np.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                                    [99., 117.],
                                                    [189., 207.],
                                                    [171., 183.]]]]).astype(np.float32)
            expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
                   name='test_conv_with_strides_and_asymmetric_padding')
        _CONV()
        _CONV_WITH_STRIDES()

layer_map['Conv'] = Conv





class Not:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Not
        self.run_ = nn._Not_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NOT():
            node = onnx.helper.make_node(
                'Not',
                inputs=['x'],
                outputs=['not'],
            )
            
            # 2d
            x = (np.random.randn(3, 4) > 0).astype(np.bool)
            expect(node, inputs=[x], outputs=[np.logical_not(x)],
                   name='test_not_2d')
            
            # 3d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            expect(node, inputs=[x], outputs=[np.logical_not(x)],
                   name='test_not_3d')
            
            # 4d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            expect(node, inputs=[x], outputs=[np.logical_not(x)],
                   name='test_not_4d')
        _NOT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Gather
        self.run_ = nn._Gather_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.data_i, self.indices_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _GATHER_0():
            node = onnx.helper.make_node(
                'Gather',
                inputs=['data', 'indices'],
                outputs=['y'],
                axis=0,
            )
            data = np.random.randn(5, 4, 3, 2).astype(np.float32)
            indices = np.array([0, 1, 3])
            y = np.take(data, indices, axis=0)
            
            expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
                   name='test_gather_0')
        def _GATHER_1():
            node = onnx.helper.make_node(
                'Gather',
                inputs=['data', 'indices'],
                outputs=['y'],
                axis=1,
            )
            data = np.random.randn(5, 4, 3, 2).astype(np.float32)
            indices = np.array([0, 1, 3])
            y = np.take(data, indices, axis=1)
            
            expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
                   name='test_gather_1')
        _GATHER_0()
        _GATHER_1()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ConvTranspose
        self.run_ = nn._ConvTranspose_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.output_padding, self.output_shape, self.pads, self.strides, self.X_i, self.W_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONVTRANSPOSE():
            x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                            [3., 4., 5.],
                            [6., 7., 8.]]]]).astype(np.float32)
            
            W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                            [1., 1., 1.],
                            [1., 1., 1.]],
                           [[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])
            
            y = np.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                            [3., 8., 15., 12., 7.],
                            [9., 21., 36., 27., 15.],
                            [9., 20., 33., 24., 13.],
                            [6., 13., 21., 15., 8.]],
            
                           [[0., 1., 3., 3., 2.],
                            [3., 8., 15., 12., 7.],
                            [9., 21., 36., 27., 15.],
                            [9., 20., 33., 24., 13.],
                            [6., 13., 21., 15., 8.]]]]).astype(np.float32)
            
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose')
        def _CONVTRANSPOSE_1D():
            x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)
            
            W = np.array([[[1., 1., 1.],  # (1, 2, 3)
                           [1., 1., 1.]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])
            
            y = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
                           [0., 1., 3., 3., 2.]]]).astype(np.float32)
            
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_1d')
        def _CONVTRANSPOSE_3D():
            x = np.array([[[[[0., 1., 2., 3., 4.],  # (1, 1, 3, 4, 5)
                             [5., 6., 7., 8., 9.],
                             [10., 11., 12., 13., 14.],
                             [15., 16., 17., 18., 19.]],
                            [[20., 21., 22., 23., 24.],
                             [25., 26., 27., 28., 29.],
                             [30., 31., 32., 33., 34.],
                             [35., 36., 37., 38., 39.]],
                            [[40., 41., 42., 43., 44.],
                             [45., 46., 47., 48., 49.],
                             [50., 51., 52., 53., 54.],
                             [55., 56., 57., 58., 59.]]]]]).astype(np.float32)
            
            W = np.array([[[[[1., 1., 1.],  # (1, 2, 3, 3, 3)
                             [1., 1., 1.],
                             [1., 1., 1.]],
                            [[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]],
                            [[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]]],
                           [[[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]],
                            [[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]],
                            [[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]]]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])
            
            y = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                             [5., 12., 21., 27., 33., 24., 13.],
                             [15., 33., 54., 63., 72., 51., 27.],
                             [30., 63., 99., 108., 117., 81., 42.],
                             [25., 52., 81., 87., 93., 64., 33.],
                             [15., 31., 48., 51., 54., 37., 19.]],
            
                            [[20., 42., 66., 72., 78., 54., 28.],
                             [50., 104., 162., 174., 186., 128., 66.],
                             [90., 186., 288., 306., 324., 222., 114.],
                             [120., 246., 378., 396., 414., 282., 144.],
                             [90., 184., 282., 294., 306., 208., 106.],
                             [50., 102., 156., 162., 168., 114., 58.]],
            
                            [[60., 123., 189., 198., 207., 141., 72.],
                             [135., 276., 423., 441., 459., 312., 159.],
                             [225., 459., 702., 729., 756., 513., 261.],
                             [270., 549., 837., 864., 891., 603., 306.],
                             [195., 396., 603., 621., 639., 432., 219.],
                             [105., 213., 324., 333., 342., 231., 117.]],
            
                            [[60., 122., 186., 192., 198., 134., 68.],
                             [130., 264., 402., 414., 426., 288., 146.],
                             [210., 426., 648., 666., 684., 462., 234.],
                             [240., 486., 738., 756., 774., 522., 264.],
                             [170., 344., 522., 534., 546., 368., 186.],
                             [90., 182., 276., 282., 288., 194., 98.]],
            
                            [[40., 81., 123., 126., 129., 87., 44.],
                             [85., 172., 261., 267., 273., 184., 93.],
                             [135., 273., 414., 423., 432., 291., 147.],
                             [150., 303., 459., 468., 477., 321., 162.],
                             [105., 212., 321., 327., 333., 224., 113.],
                             [55., 111., 168., 171., 174., 117., 59.]]],
            
                           [[[0., 1., 3., 6., 9., 7., 4.],
                             [5., 12., 21., 27., 33., 24., 13.],
                             [15., 33., 54., 63., 72., 51., 27.],
                             [30., 63., 99., 108., 117., 81., 42.],
                             [25., 52., 81., 87., 93., 64., 33.],
                             [15., 31., 48., 51., 54., 37., 19.]],
            
                            [[20., 42., 66., 72., 78., 54., 28.],
                             [50., 104., 162., 174., 186., 128., 66.],
                             [90., 186., 288., 306., 324., 222., 114.],
                             [120., 246., 378., 396., 414., 282., 144.],
                             [90., 184., 282., 294., 306., 208., 106.],
                             [50., 102., 156., 162., 168., 114., 58.]],
            
                            [[60., 123., 189., 198., 207., 141., 72.],
                             [135., 276., 423., 441., 459., 312., 159.],
                             [225., 459., 702., 729., 756., 513., 261.],
                             [270., 549., 837., 864., 891., 603., 306.],
                             [195., 396., 603., 621., 639., 432., 219.],
                             [105., 213., 324., 333., 342., 231., 117.]],
            
                            [[60., 122., 186., 192., 198., 134., 68.],
                             [130., 264., 402., 414., 426., 288., 146.],
                             [210., 426., 648., 666., 684., 462., 234.],
                             [240., 486., 738., 756., 774., 522., 264.],
                             [170., 344., 522., 534., 546., 368., 186.],
                             [90., 182., 276., 282., 288., 194., 98.]],
            
                            [[40., 81., 123., 126., 129., 87., 44.],
                             [85., 172., 261., 267., 273., 184., 93.],
                             [135., 273., 414., 423., 432., 291., 147.],
                             [150., 303., 459., 468., 477., 321., 162.],
                             [105., 212., 321., 327., 333., 224., 113.],
                             [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(np.float32)
            
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_3d')
        def _CONVTRANSPOSE_ATTRIBUTES():
            x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                            [3., 4., 5.],
                            [6., 7., 8.]]]]).astype(np.float32)
            
            W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                            [1., 1., 1.],
                            [1., 1., 1.]],
                           [[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]]]]).astype(np.float32)
            
            y = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                            [0., 0., 1., 1., 3., 2., 2., 0.],
                            [0., 0., 1., 1., 3., 2., 2., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0.]],
            
                           [[0., 0., 1., 1., 3., 2., 2., 0.],
                            [0., 0., 1., 1., 3., 2., 2., 0.],
                            [0., 0., 1., 1., 3., 2., 2., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [3., 3., 7., 4., 9., 5., 5., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [6., 6., 13., 7., 15., 8., 8., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         strides=[3, 2],
                                         output_shape=[10, 8])
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_output_shape')
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         strides=[3, 2],
                                         output_padding=[1, 1])
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pad')
            
            node = onnx.helper.make_node(
                'ConvTranspose', ['X', 'W'], ['Y'],
                name='test',
                strides=[3, 2],
                output_shape=[10, 8],
                kernel_shape=[3, 3],
                output_padding=[1, 1]
            )
            expect(node, inputs=[x, W], outputs=[y],
                   name='test_convtranspose_kernel_shape')
        def _CONVTRANSPOSE_PADS():
            x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                            [3., 4., 5.],
                            [6., 7., 8.]]]]).astype(np.float32)
            
            W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                            [1., 1., 1.],
                            [1., 1., 1.]],
                           [[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         strides=[3, 2],
                                         pads=[1, 2, 1, 2])
            
            y = np.array([[[[1., 1., 3.],  # (1, 2, 7, 3)
                            [1., 1., 3.],
                            [7., 4., 9.],
                            [7., 4., 9.],
                            [7., 4., 9.],
                            [13., 7., 15.],
                            [13., 7., 15.]],
            
                           [[1., 1., 3.],
                            [1., 1., 3.],
                            [7., 4., 9.],
                            [7., 4., 9.],
                            [7., 4., 9.],
                            [13., 7., 15.],
                            [13., 7., 15.]]]]).astype(np.float32)
            
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pads')
        def _CONVTRANSPOSE_DILATIONS():
            x = np.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                            [9., 5., 7.],
                            [3., 2., 6.]]]]).astype(np.float32)
            W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
                            [1., 9.]]]]).astype(np.float32)
            
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2])
            
            y = np.array([[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                            [63., 35., 67., 10., 14.],
                            [24., 22., 76., 76., 21.],
                            [9., 5., 88., 45., 63.],
                            [3., 2., 33., 18., 54.]]]]).astype(np.float32)
            
            expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_dilations')
        _CONVTRANSPOSE()
        _CONVTRANSPOSE_1D()
        _CONVTRANSPOSE_3D()
        _CONVTRANSPOSE_ATTRIBUTES()
        _CONVTRANSPOSE_PADS()
        _CONVTRANSPOSE_DILATIONS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Dropout
        self.run_ = nn._Dropout_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.ratio, self.data_i, self.output_o, self.mask_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEFAULT():
            node = onnx.helper.make_node(
                'Dropout',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = x
            expect(node, inputs=[x], outputs=[y],
                   name='test_dropout_default')
        def _RANDOM():
            node = onnx.helper.make_node(
                'Dropout',
                inputs=['x'],
                outputs=['y'],
                ratio=.2,
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = x
            expect(node, inputs=[x], outputs=[y],
                   name='test_dropout_random')
        _DEFAULT()
        _RANDOM()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LeakyRelu
        self.run_ = nn._LeakyRelu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _LEAKYRELU():
            node = onnx.helper.make_node(
                'LeakyRelu',
                inputs=['x'],
                outputs=['y'],
                alpha=0.1
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            # expected output [-0.1, 0., 1.]
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
            expect(node, inputs=[x], outputs=[y],
                   name='test_leakyrelu_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
            expect(node, inputs=[x], outputs=[y],
                   name='test_leakyrelu')
        def _LEAKYRELU_DEFAULT():
            default_alpha = 0.01
            node = onnx.helper.make_node(
                'LeakyRelu',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * default_alpha
            expect(node, inputs=[x], outputs=[y],
                   name='test_leakyrelu_default')
        _LEAKYRELU()
        _LEAKYRELU_DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Elu
        self.run_ = nn._Elu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ELU():
            node = onnx.helper.make_node(
                'Elu',
                inputs=['x'],
                outputs=['y'],
                alpha=2.0
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            # expected output [-1.2642411, 0., 1.]
            y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
            expect(node, inputs=[x], outputs=[y],
                   name='test_elu_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
            expect(node, inputs=[x], outputs=[y],
                   name='test_elu')
        def _ELU_DEFAULT():
            default_alpha = 1.0
            node = onnx.helper.make_node(
                'Elu',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
            expect(node, inputs=[x], outputs=[y],
                   name='test_elu_default')
        _ELU()
        _ELU_DEFAULT()

layer_map['Elu'] = Elu





class GlobalAveragePool:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._GlobalAveragePool
        self.run_ = nn._GlobalAveragePool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _GLOBALAVERAGEPOOL():
            node = onnx.helper.make_node(
                'GlobalAveragePool',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(1, 3, 5, 5).astype(np.float32)
            spatial_shape = np.ndim(x) - 2
            y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
            for _ in range(spatial_shape):
                y = np.expand_dims(y, -1)
            expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool')
        def _GLOBALAVERAGEPOOL_PRECOMPUTED():
            
            node = onnx.helper.make_node(
                'GlobalAveragePool',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.array([[[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]]]).astype(np.float32)
            y = np.array([[[[5]]]]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool_precomputed')
        _GLOBALAVERAGEPOOL()
        _GLOBALAVERAGEPOOL_PRECOMPUTED()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Gemm
        self.run_ = nn._Gemm_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.beta, self.transA, self.transB, self.A_i, self.B_i, self.C_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TRANSPOSE():
            node = onnx.helper.make_node(
                'Gemm',
                inputs=['a', 'b', 'c'],
                outputs=['y'],
                alpha=0.5,
                beta=0.5,
                transA=1,
                transB=1
            )
            a = np.random.ranf([6, 3]).astype(np.float32)
            b = np.random.ranf([4, 6]).astype(np.float32)
            c = np.random.ranf([1, 1]).astype(np.float32)
            y = 0.5 * np.dot(a.T, b.T) + 0.5 * c
            expect(node, inputs=[a, b, c], outputs=[y],
                   name='test_gemm_broadcast')
        def _NOTRANSPOSE():
            node = onnx.helper.make_node(
                'Gemm',
                inputs=['a', 'b', 'c'],
                outputs=['y'],
                alpha=0.5,
                beta=0.5
            )
            a = np.random.ranf([3, 6]).astype(np.float32)
            b = np.random.ranf([6, 4]).astype(np.float32)
            c = np.random.ranf([3, 4]).astype(np.float32)
            y = 0.5 * np.dot(a, b) + 0.5 * c
            expect(node, inputs=[a, b, c], outputs=[y],
                   name='test_gemm_nobroadcast')
        _TRANSPOSE()
        _NOTRANSPOSE()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MaxPool
        self.run_ = nn._MaxPool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.kernel_shape, self.auto_pad, self.ceil_mode, self.dilations, self.pads, self.storage_order, self.strides, self.X_i, self.Y_o, self.Indices_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MAXPOOL_2D_PRECOMPUTED_PADS():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 5, 5]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2]
            
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[
                [13, 14, 15, 15, 15],
                [18, 19, 20, 20, 20],
                [23, 24, 25, 25, 25],
                [23, 24, 25, 25, 25],
                [23, 24, 25, 25, 25]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_pads')
        def _MAXPOOL_WITH_ARGMAX_2D_PRECOMPUTED_PADS():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 5, 5]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y', 'z'],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2]
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[
                [13, 14, 15, 15, 15],
                [18, 19, 20, 20, 20],
                [23, 24, 25, 25, 25],
                [23, 24, 25, 25, 25],
                [23, 24, 25, 25, 25]]]]).astype(np.float32)
            z = np.array([[[
                [12, 13, 14, 14, 14],
                [17, 18, 19, 19, 19],
                [22, 23, 24, 24, 24],
                [22, 23, 24, 24, 24],
                [22, 23, 24, 24, 24]]]]).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_pads')
        def _MAXPOOL_2D_PRECOMPUTED_STRIDES():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[7, 9],
                            [17, 19]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_strides')
        def _MAXPOOL_WITH_ARGMAX_2D_PRECOMPUTED_STRIDES():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y', 'z'],
                kernel_shape=[2, 2],
                strides=[2, 2],
                storage_order=1
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[7, 9],
                            [17, 19]]]]).astype(np.float32)
            z = np.array([[[[6, 16],
                            [8, 18]]]]).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_strides')
        def _MAXPOOL_2D_PRECOMPUTED_SAME_UPPER():
            """
            input_shape: [1, 1, 5, 5]
            output_shape: [1, 1, 3, 3]
            pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.array([[[
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]]]).astype(np.float32)
            y = np.array([[[[7, 9, 10],
                            [17, 19, 20],
                            [22, 24, 25]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_same_upper')
        def _MAXPOOL_1D_DEFAULT():
            """
            input_shape: [1, 3, 32]
            output_shape: [1, 3, 31]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2],
            )
            x = np.random.randn(1, 3, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2]
            strides = [1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_1d_default')
        def _MAXPOOL_2D_DEFAULT():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 31, 31]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_default')
        def _MAXPOOL_3D_DEFAULT():
            """
            input_shape: [1, 3, 32, 32, 32]
            output_shape: [1, 3, 31, 31, 31]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2, 2],
            )
            x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = [2, 2, 2]
            strides = [1, 1, 1]
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_default')
        def _MAXPOOL_2D_SAME_UPPER():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 32, 32]
            pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_UPPER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_top = pad_shape[0] // 2
            pad_bottom = pad_shape[0] - pad_top
            pad_left = pad_shape[1] // 2
            pad_right = pad_shape[1] - pad_left
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_upper')
        def _MAXPOOL_2D_SAME_LOWER():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 32, 32]
            pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                auto_pad='SAME_LOWER'
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (2, 2)
            strides = (1, 1)
            out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
            pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
            pad_bottom = pad_shape[0] // 2
            pad_top = pad_shape[0] - pad_bottom
            pad_right = pad_shape[1] // 2
            pad_left = pad_shape[1] - pad_right
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_lower')
        def _MAXPOOL_2D_PADS():
            """
            input_shape: [1, 3, 28, 28]
            output_shape: [1, 3, 30, 30]
            pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[2, 2, 2, 2]
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (1, 1)
            pad_bottom = pad_top = pad_right = pad_left = 2
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                            constant_values=np.nan)
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_pads')
        def _MAXPOOL_2D_STRIDES():
            """
            input_shape: [1, 3, 32, 32]
            output_shape: [1, 3, 10, 10]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[5, 5],
                strides=[3, 3]
            )
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (5, 5)
            strides = (3, 3)
            out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
            padded = x
            y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')
        def _MAXPOOL_2D_CEIL():
            """
            input_shape: [1, 1, 4, 4]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=True
            )
            x = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]]]).astype(np.float32)
            y = np.array([[[
                [11, 12],
                [15, 16]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_ceil')
        def _MAXPOOL_2D_DILATIONS():
            """
            input_shape: [1, 1, 4, 4]
            output_shape: [1, 1, 2, 2]
            """
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[1, 1],
                dilations=[2, 2]
            )
            x = np.array([[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]]]).astype(np.float32)
            y = np.array([[[
                [11, 12],
                [15, 16]]]]).astype(np.float32)
            
            expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_dilations')
        _MAXPOOL_2D_PRECOMPUTED_PADS()
        _MAXPOOL_WITH_ARGMAX_2D_PRECOMPUTED_PADS()
        _MAXPOOL_2D_PRECOMPUTED_STRIDES()
        _MAXPOOL_WITH_ARGMAX_2D_PRECOMPUTED_STRIDES()
        _MAXPOOL_2D_PRECOMPUTED_SAME_UPPER()
        _MAXPOOL_1D_DEFAULT()
        _MAXPOOL_2D_DEFAULT()
        _MAXPOOL_3D_DEFAULT()
        _MAXPOOL_2D_SAME_UPPER()
        _MAXPOOL_2D_SAME_LOWER()
        _MAXPOOL_2D_PADS()
        _MAXPOOL_2D_STRIDES()
        _MAXPOOL_2D_CEIL()
        _MAXPOOL_2D_DILATIONS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Equal
        self.run_ = nn._Equal_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _EQUAL():
            node = onnx.helper.make_node(
                'Equal',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
            y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
            z = np.equal(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_equal')
        def _EQUAL_BROADCAST():
            node = onnx.helper.make_node(
                'Equal',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
            y = (np.random.randn(5) * 10).astype(np.int32)
            z = np.equal(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_equal_bcast')
        _EQUAL()
        _EQUAL_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Tile
        self.run_ = nn._Tile_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.repeats_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TILE():
            node = onnx.helper.make_node(
                'Tile',
                inputs=['x', 'y'],
                outputs=['z']
            )
            
            x = np.random.rand(2, 3, 4, 5).astype(np.float32)
            
            repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
            
            z = np.tile(x, repeats)
            
            expect(node,
                   inputs=[x, repeats],
                   outputs=[z],
                   name='test_tile')
        def _TILE_PRECOMPUTED():
            node = onnx.helper.make_node(
                'Tile',
                inputs=['x', 'y'],
                outputs=['z']
            )
            
            x = np.array([
                [0, 1],
                [2, 3]
            ], dtype=np.float32)
            
            repeats = np.array([2, 2], dtype=np.int64)
            
            z = np.array([
                [0, 1, 0, 1],
                [2, 3, 2, 3],
                [0, 1, 0, 1],
                [2, 3, 2, 3]
            ], dtype=np.float32)
            
            expect(node,
                   inputs=[x, repeats],
                   outputs=[z],
                   name='test_tile_precomputed')
        _TILE()
        _TILE_PRECOMPUTED()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Flatten
        self.run_ = nn._Flatten_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _FLATTEN():
            shape = (2, 3, 4, 5)
            a = np.random.random_sample(shape).astype(np.float32)
            
            for i in range(len(shape)):
                node = onnx.helper.make_node(
                    'Flatten',
                    inputs=['a'],
                    outputs=['b'],
                    axis=i,
                )
            
                new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
                b = np.reshape(a, new_shape)
                expect(node, inputs=[a], outputs=[b],
                       name='test_flatten_axis' + str(i))
        def _FLATTEN_WITH_DEFAULT_AXIS():
            node = onnx.helper.make_node(
                'Flatten',
                inputs=['a'],
                outputs=['b'],  # Default value for axis: axis=1
            )
            
            shape = (5, 4, 3, 2)
            a = np.random.random_sample(shape).astype(np.float32)
            new_shape = (5, 24)
            b = np.reshape(a, new_shape)
            expect(node, inputs=[a], outputs=[b],
                   name='test_flatten_default_axis')
        _FLATTEN()
        _FLATTEN_WITH_DEFAULT_AXIS()

layer_map['Flatten'] = Flatten





class Floor:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Floor
        self.run_ = nn._Floor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _FLOOR():
            node = onnx.helper.make_node(
                'Floor',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1.5, 1.2, 2]).astype(np.float32)
            y = np.floor(x)  # expected output [-2., 1., 2.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_floor_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.floor(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_floor')
        _FLOOR()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._GRU
        self.run_ = nn._GRU_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.activation_alpha, self.activation_beta, self.activations, self.clip, self.direction, self.hidden_size, self.linear_before_reset, self.X_i, self.W_i, self.R_i, self.B_i, self.sequence_lens_i, self.initial_h_i, self.Y_o, self.Y_h_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEFAULTS():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 5
            weight_scale = 0.1
            number_of_gates = 3
            
            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            
            gru = GRU_Helper(X=input, W=W, R=R)
            _, Y_h = gru.step()
            expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_gru_defaults')
        def _INITIAL_BIAS():
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)
            
            input_size = 3
            hidden_size = 3
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 3
            
            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            
            # Adding custom bias
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)
            
            gru = GRU_Helper(X=input, W=W, R=R, B=B)
            _, Y_h = gru.step()
            expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_with_initial_bias')
        def _SEQ_LENGTH():
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                              [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)
            
            input_size = 3
            hidden_size = 5
            number_of_gates = 3
            
            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['', 'Y'],
                hidden_size=hidden_size
            )
            
            W = np.random.randn(1, number_of_gates * hidden_size, input_size).astype(np.float32)
            R = np.random.randn(1, number_of_gates * hidden_size, hidden_size).astype(np.float32)
            
            # Adding custom bias
            W_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
            R_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)
            
            gru = GRU_Helper(X=input, W=W, R=R, B=B)
            _, Y_h = gru.step()
            expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_seq_length')
        _DEFAULTS()
        _INITIAL_BIAS()
        _SEQ_LENGTH()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._GlobalLpPool
        self.run_ = nn._GlobalLpPool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.p, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Greater
        self.run_ = nn._Greater_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _GREATER():
            node = onnx.helper.make_node(
                'Greater',
                inputs=['x', 'y'],
                outputs=['greater'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            z = np.greater(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_greater')
        def _GREATER_BROADCAST():
            node = onnx.helper.make_node(
                'Greater',
                inputs=['x', 'y'],
                outputs=['greater'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(5).astype(np.float32)
            z = np.greater(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_greater_bcast')
        _GREATER()
        _GREATER_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._HardSigmoid
        self.run_ = nn._HardSigmoid_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.beta, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _HARDSIGMOID():
            node = onnx.helper.make_node(
                'HardSigmoid',
                inputs=['x'],
                outputs=['y'],
                alpha=0.5,
                beta=0.6
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.clip(x * 0.5 + 0.6, 0, 1)  # expected output [0.1, 0.6, 1.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardsigmoid_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x * 0.5 + 0.6, 0, 1)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardsigmoid')
        def _HARDSIGMOID_DEFAULT():
            default_alpha = 0.2
            default_beta = 0.5
            node = onnx.helper.make_node(
                'HardSigmoid',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x * default_alpha + default_beta, 0, 1)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardsigmoid_default')
        _HARDSIGMOID()
        _HARDSIGMOID_DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Selu
        self.run_ = nn._Selu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.gamma, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SELU():
            node = onnx.helper.make_node(
                'Selu',
                inputs=['x'],
                outputs=['y'],
                alpha=2.0,
                gamma=3.0
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            # expected output [-3.79272318, 0., 3.]
            y = np.clip(x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
            expect(node, inputs=[x], outputs=[y],
                   name='test_selu_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
            expect(node, inputs=[x], outputs=[y],
                   name='test_selu')
        def _SELU_DEFAULT():
            default_alpha = 1.67326319217681884765625
            default_gamma = 1.05070102214813232421875
            node = onnx.helper.make_node(
                'Selu',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) * default_gamma + \
                (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
            expect(node, inputs=[x], outputs=[y],
                   name='test_selu_default')
        _SELU()
        _SELU_DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Hardmax
        self.run_ = nn._Hardmax_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _HARDMAX():
            node = onnx.helper.make_node(
                'Hardmax',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
            y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_example')
            
            # For multiple occurrances of the maximal values, the first occurrence is selected for one-hot output
            x = np.array([[3, 3, 3, 1]]).astype(np.float32)
            y = np.array([[1, 0, 0, 0]]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_one_hot')
        def _HARDMAX_AXIS():
            def hardmax_2d(x):  # type: (np.ndarray) -> np.ndarray
                return np.eye(x.shape[1], dtype=x.dtype)[np.argmax(x, axis=1)]
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            node = onnx.helper.make_node(
                'Hardmax',
                inputs=['x'],
                outputs=['y'],
                axis=0,
            )
            y = hardmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_axis_0')
            
            node = onnx.helper.make_node(
                'Hardmax',
                inputs=['x'],
                outputs=['y'],
                axis=1,
            )
            y = hardmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_axis_1')
            
            # default axis is 1
            node = onnx.helper.make_node(
                'Hardmax',
                inputs=['x'],
                outputs=['y'],
            )
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_default_axis')
            
            node = onnx.helper.make_node(
                'Hardmax',
                inputs=['x'],
                outputs=['y'],
                axis=2,
            )
            y = hardmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_hardmax_axis_2')
        _HARDMAX()
        _HARDMAX_AXIS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._If
        self.run_ = nn._If_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.else_branch, self.then_branch, self.cond_i)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['If'] = If





class Min:
    name = None
    min_o = str()

    #parameters

    input_params = []
    output_params = ["min_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Min
        self.run_ = nn._Min_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.min_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MIN():
            data_0 = np.array([3, 2, 1]).astype(np.float32)
            data_1 = np.array([1, 4, 4]).astype(np.float32)
            data_2 = np.array([2, 5, 0]).astype(np.float32)
            result = np.array([1, 2, 0]).astype(np.float32)
            node = onnx.helper.make_node(
                'Min',
                inputs=['data_0', 'data_1', 'data_2'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
                   name='test_min_example')
            
            node = onnx.helper.make_node(
                'Min',
                inputs=['data_0'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0], outputs=[data_0],
                   name='test_min_one_input')
            
            result = np.minimum(data_0, data_1)
            node = onnx.helper.make_node(
                'Min',
                inputs=['data_0', 'data_1'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1], outputs=[result],
                   name='test_min_two_inputs')
        _MIN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._InstanceNormalization
        self.run_ = nn._InstanceNormalization_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.epsilon, self.input_i, self.scale_i, self.B_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _INSTANCENORMALIZATION():
            def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
                dims_x = len(x.shape)
                axis = tuple(range(2, dims_x))
                mean = np.mean(x, axis=axis, keepdims=True)
                var = np.var(x, axis=axis, keepdims=True)
                dim_ones = (1,) * (dims_x - 2)
                s = s.reshape(-1, *dim_ones)
                bias = bias.reshape(-1, *dim_ones)
                return s * (x - mean) / np.sqrt(var + epsilon) + bias
            
            # input size: (1, 2, 1, 3)
            x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
            s = np.array([1.0, 1.5]).astype(np.float32)
            bias = np.array([0, 1]).astype(np.float32)
            y = _instancenorm_test_mode(x, s, bias).astype(np.float32)
            
            node = onnx.helper.make_node(
                'InstanceNormalization',
                inputs=['x', 's', 'bias'],
                outputs=['y'],
            )
            
            # output size: (1, 2, 1, 3)
            expect(node, inputs=[x, s, bias], outputs=[y],
                   name='test_instancenorm_example')
            
            # input size: (2, 3, 4, 5)
            x = np.random.randn(2, 3, 4, 5).astype(np.float32)
            s = np.random.randn(3).astype(np.float32)
            bias = np.random.randn(3).astype(np.float32)
            epsilon = 1e-2
            y = _instancenorm_test_mode(x, s, bias, epsilon).astype(np.float32)
            
            node = onnx.helper.make_node(
                'InstanceNormalization',
                inputs=['x', 's', 'bias'],
                outputs=['y'],
                epsilon=epsilon,
            )
            
            # output size: (2, 3, 4, 5)
            expect(node, inputs=[x, s, bias], outputs=[y],
                   name='test_instancenorm_epsilon')
        _INSTANCENORMALIZATION()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Less
        self.run_ = nn._Less_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _LESS():
            node = onnx.helper.make_node(
                'Less',
                inputs=['x', 'y'],
                outputs=['less'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            z = np.less(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_less')
        def _LESS_BROADCAST():
            node = onnx.helper.make_node(
                'Less',
                inputs=['x', 'y'],
                outputs=['less'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(5).astype(np.float32)
            z = np.less(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_less_bcast')
        _LESS()
        _LESS_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._EyeLike
        self.run_ = nn._EyeLike_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.dtype, self.k, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _WITHOUT_DTYPE():
            shape = (4, 4)
            node = onnx.helper.make_node(
                'EyeLike',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.random.randint(0, 100, size=shape, dtype=np.int32)
            y = np.eye(shape[0], shape[1], dtype=np.int32)
            expect(node, inputs=[x], outputs=[y], name='test_eyelike_without_dtype')
        def _WITH_DTYPE():
            shape = (3, 4)
            node = onnx.helper.make_node(
                'EyeLike',
                inputs=['x'],
                outputs=['y'],
                dtype=onnx.TensorProto.DOUBLE,
            )
            
            x = np.random.randint(0, 100, size=shape, dtype=np.int32)
            y = np.eye(shape[0], shape[1], dtype=np.float64)
            expect(node, inputs=[x], outputs=[y], name='test_eyelike_with_dtype')
        def _POPULATE_OFF_MAIN_DIAGONAL():
            shape = (4, 5)
            off_diagonal_offset = 1
            node = onnx.helper.make_node(
                'EyeLike',
                inputs=['x'],
                outputs=['y'],
                k=off_diagonal_offset,
                dtype=onnx.TensorProto.FLOAT,
            )
            
            x = np.random.randint(0, 100, size=shape, dtype=np.int32)
            y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
            expect(node, inputs=[x], outputs=[y], name='test_eyelike_populate_off_main_diagonal')
        _WITHOUT_DTYPE()
        _WITH_DTYPE()
        _POPULATE_OFF_MAIN_DIAGONAL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RandomNormal
        self.run_ = nn._RandomNormal_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.shape, self.dtype, self.mean, self.scale, self.seed, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Slice
        self.run_ = nn._Slice_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.data_i, self.starts_i, self.ends_i, self.axes_i, self.steps_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SLICE():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes', 'steps'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            y = x[0:3, 0:10]
            starts = np.array([0, 0], dtype=np.int64)
            ends = np.array([3, 10], dtype=np.int64)
            axes = np.array([0, 1], dtype=np.int64)
            steps = np.array([1, 1], dtype=np.int64)
            
            expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
                   name='test_slice')
        def _SLICE_NEG():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes', 'steps'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([0], dtype=np.int64)
            ends = np.array([-1], dtype=np.int64)
            axes = np.array([1], dtype=np.int64)
            steps = np.array([1], dtype=np.int64)
            y = x[:, 0:-1]
            
            expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
                   name='test_slice_neg')
        def _SLICE_START_OUT_OF_BOUNDS():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes', 'steps'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([1000], dtype=np.int64)
            ends = np.array([1000], dtype=np.int64)
            axes = np.array([1], dtype=np.int64)
            steps = np.array([1], dtype=np.int64)
            y = x[:, 1000:1000]
            
            expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
                   name='test_slice_start_out_of_bounds')
        def _SLICE_END_OUT_OF_BOUNDS():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes', 'steps'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([1], dtype=np.int64)
            ends = np.array([1000], dtype=np.int64)
            axes = np.array([1], dtype=np.int64)
            steps = np.array([1], dtype=np.int64)
            y = x[:, 1:1000]
            
            expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
                   name='test_slice_end_out_of_bounds')
        def _SLICE_DEFAULT_AXES():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([0, 0, 3], dtype=np.int64)
            ends = np.array([20, 10, 4], dtype=np.int64)
            y = x[:, :, 3:4]
            
            expect(node, inputs=[x, starts, ends], outputs=[y],
                   name='test_slice_default_axes')
        def _SLICE_DEFAULT_STEPS():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([0, 0, 3], dtype=np.int64)
            ends = np.array([20, 10, 4], dtype=np.int64)
            axes = np.array([0, 1, 2], dtype=np.int64)
            y = x[:, :, 3:4]
            
            expect(node, inputs=[x, starts, ends, axes], outputs=[y],
                   name='test_slice_default_steps')
        def _SLICE_NEG_STEPS():
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x', 'starts', 'ends', 'axes', 'steps'],
                outputs=['y'],
            )
            
            x = np.random.randn(20, 10, 5).astype(np.float32)
            starts = np.array([20, 10, 4], dtype=np.int64)
            ends = np.array([0, 0, 1], dtype=np.int64)
            axes = np.array([0, 1, 2], dtype=np.int64)
            steps = np.array([-1, -3, -2])
            y = x[20:0:-1, 10:0:-3, 4:1:-2]
            
            expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
                   name='test_slice_neg_steps')
        _SLICE()
        _SLICE_NEG()
        _SLICE_START_OUT_OF_BOUNDS()
        _SLICE_END_OUT_OF_BOUNDS()
        _SLICE_DEFAULT_AXES()
        _SLICE_DEFAULT_STEPS()
        _SLICE_NEG_STEPS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._PRelu
        self.run_ = nn._PRelu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.slope_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _PRELU():
            node = onnx.helper.make_node(
                'PRelu',
                inputs=['x', 'slope'],
                outputs=['y'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            slope = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
            
            expect(node, inputs=[x, slope], outputs=[y],
                   name='test_prelu_example')
        def _PRELU_BROADCAST():
            node = onnx.helper.make_node(
                'PRelu',
                inputs=['x', 'slope'],
                outputs=['y'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            slope = np.random.randn(5).astype(np.float32)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
            
            expect(node, inputs=[x, slope], outputs=[y],
                   name='test_prelu_broadcast')
        _PRELU()
        _PRELU_BROADCAST()

layer_map['PRelu'] = PRelu





class Log:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Log
        self.run_ = nn._Log_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _LOG():
            node = onnx.helper.make_node(
                'Log',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([1, 10]).astype(np.float32)
            y = np.log(x)  # expected output [0., 2.30258512]
            expect(node, inputs=[x], outputs=[y],
                   name='test_log_example')
            
            x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
            y = np.log(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_log')
        _LOG()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LogSoftmax
        self.run_ = nn._LogSoftmax_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _LOGSOFTMAX():
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.array([[-1, 0, 1]]).astype(np.float32)
            # expected output [[-2.40760589, -1.40760589, -0.40760589]]
            y = x - np.log(np.sum(np.exp(x), axis=1))
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_example_1')
        def _LOGSOFTMAX_AXIS():
            def logsoftmax_2d(x):  # type: (np.ndarray) -> np.ndarray
                max_x = np.max(x, axis=1).reshape((-1, 1))
                exp_x = np.exp(x - max_x)
                return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))
            
            x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
            # expected output [[-3.4401896, -2.4401896, -1.44018972, -0.44018969],
            #                 [-3.4401896, -2.4401896, -1.44018972, -0.44018969]]
            y = logsoftmax_2d(x)
            
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
            )
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_large_number')
            
            x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
                axis=0,
            )
            y = logsoftmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_axis_0')
            
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
                axis=1,
            )
            y = logsoftmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_axis_1')
            
            # default axis is 1
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
            )
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_default_axis')
            
            node = onnx.helper.make_node(
                'LogSoftmax',
                inputs=['x'],
                outputs=['y'],
                axis=2,
            )
            y = logsoftmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_logsoftmax_axis_2')
        _LOGSOFTMAX()
        _LOGSOFTMAX_AXIS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Loop
        self.run_ = nn._Loop_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.body, self.M_i, self.cond_i)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LpNormalization
        self.run_ = nn._LpNormalization_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.p, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MatMul
        self.run_ = nn._MatMul_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MATMUL():
            node = onnx.helper.make_node(
                'MatMul',
                inputs=['a', 'b'],
                outputs=['c'],
            )
            
            # 2d
            a = np.random.randn(3, 4).astype(np.float32)
            b = np.random.randn(4, 3).astype(np.float32)
            c = np.matmul(a, b)
            expect(node, inputs=[a, b], outputs=[c],
                   name='test_matmul_2d')
            
            # 3d
            a = np.random.randn(2, 3, 4).astype(np.float32)
            b = np.random.randn(2, 4, 3).astype(np.float32)
            c = np.matmul(a, b)
            expect(node, inputs=[a, b], outputs=[c],
                   name='test_matmul_3d')
            
            # 4d
            a = np.random.randn(1, 2, 3, 4).astype(np.float32)
            b = np.random.randn(1, 2, 4, 3).astype(np.float32)
            c = np.matmul(a, b)
            expect(node, inputs=[a, b], outputs=[c],
                   name='test_matmul_4d')
        _MATMUL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceL2
        self.run_ = nn._ReduceL2_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [2]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceL2',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
            #print(reduced)
            #[[2.23606798, 5.],
            # [7.81024968, 10.63014581],
            # [13.45362405, 16.2788206]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l2_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l2_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [2]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceL2',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
            #print(reduced)
            #[[[2.23606798], [5.]]
            # [[7.81024968], [10.63014581]]
            # [[13.45362405], [16.2788206 ]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l2_keep_dims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_l2_keep_dims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceL2',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=axes, keepdims=keepdims == 1))
            #print(reduced)
            #[[[25.49509757]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l2_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sqrt(np.sum(
                a=np.square(data), axis=axes, keepdims=keepdims == 1))
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l2_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

layer_map['ReduceL2'] = ReduceL2





class Max:
    name = None
    max_o = str()

    #parameters

    input_params = []
    output_params = ["max_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Max
        self.run_ = nn._Max_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.max_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MAX():
            data_0 = np.array([3, 2, 1]).astype(np.float32)
            data_1 = np.array([1, 4, 4]).astype(np.float32)
            data_2 = np.array([2, 5, 3]).astype(np.float32)
            result = np.array([3, 5, 4]).astype(np.float32)
            node = onnx.helper.make_node(
                'Max',
                inputs=['data_0', 'data_1', 'data_2'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
                   name='test_max_example')
            
            node = onnx.helper.make_node(
                'Max',
                inputs=['data_0'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0], outputs=[data_0],
                   name='test_max_one_input')
            
            result = np.maximum(data_0, data_1)
            node = onnx.helper.make_node(
                'Max',
                inputs=['data_0', 'data_1'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1], outputs=[result],
                   name='test_max_two_inputs')
        _MAX()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MaxRoiPool
        self.run_ = nn._MaxRoiPool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.pooled_shape, self.spatial_scale, self.X_i, self.rois_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Or
        self.run_ = nn._Or_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _OR():
            node = onnx.helper.make_node(
                'Or',
                inputs=['x', 'y'],
                outputs=['or'],
            )
            
            # 2d
            x = (np.random.randn(3, 4) > 0).astype(np.bool)
            y = (np.random.randn(3, 4) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or2d')
            
            # 3d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or3d')
            
            # 4d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or4d')
        def _OR_BROADCAST():
            node = onnx.helper.make_node(
                'Or',
                inputs=['x', 'y'],
                outputs=['or'],
            )
            
            # 3d vs 1d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(5) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or_bcast3v1d')
            
            # 3d vs 2d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(4, 5) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or_bcast3v2d')
            
            # 4d vs 2d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(5, 6) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or_bcast4v2d')
            
            # 4d vs 3d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or_bcast4v3d')
            
            # 4d vs 4d
            x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
            z = np.logical_or(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_or_bcast4v4d')
        _OR()
        _OR_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Pad
        self.run_ = nn._Pad_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.pads, self.mode, self.value, self.data_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONSTANT_PAD():
            node = onnx.helper.make_node(
                'Pad',
                inputs=['x'],
                outputs=['y'],
                mode='constant',
                value=1.2,
                pads=[0, 0, 1, 3, 0, 0, 2, 4],
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.float32)
            y = np.pad(
                x,
                pad_width=((0, 0), (0, 0), (1, 2), (3, 4)),
                mode='constant',
                constant_values=1.2,
            )
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_constant_pad')
        def _REFLECTION_AND_EDGE_PAD():
            for mode in ['edge', 'reflect']:
                node = onnx.helper.make_node(
                    'Pad',
                    inputs=['x'],
                    outputs=['y'],
                    mode=mode,
                    pads=[0, 0, 1, 1, 0, 0, 1, 1]
                )
                x = np.random.randn(1, 3, 4, 5).astype(np.float32)
                y = np.pad(
                    x,
                    pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode=mode,
                )
            
                expect(node, inputs=[x], outputs=[y],
                       name='test_{}_pad'.format(mode))
        _CONSTANT_PAD()
        _REFLECTION_AND_EDGE_PAD()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RandomUniformLike
        self.run_ = nn._RandomUniformLike_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.dtype, self.high, self.low, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['RandomUniformLike'] = RandomUniformLike





class Reciprocal:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Reciprocal
        self.run_ = nn._Reciprocal_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _RECIPROCAL():
            node = onnx.helper.make_node(
                'Reciprocal',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-4, 2]).astype(np.float32)
            y = np.reciprocal(x)  # expected output [-0.25, 0.5],
            expect(node, inputs=[x], outputs=[y],
                   name='test_reciprocal_example')
            
            x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
            y = np.reciprocal(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_reciprocal')
        _RECIPROCAL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Pow
        self.run_ = nn._Pow_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_i, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _POW():
            node = onnx.helper.make_node(
                'Pow',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([1, 2, 3]).astype(np.float32)
            y = np.array([4, 5, 6]).astype(np.float32)
            z = np.power(x, y)  # expected output [1., 32., 729.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_pow_example')
            
            x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            z = np.power(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_pow')
        def _POW_BROADCAST():
            node = onnx.helper.make_node(
                'Pow',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([1, 2, 3]).astype(np.float32)
            y = np.array(2).astype(np.float32)
            z = np.power(x, y)  # expected output [1., 4., 9.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_pow_bcast_scalar')
            
            node = onnx.helper.make_node(
                'Pow',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            y = np.array([1, 2, 3]).astype(np.float32)
            # expected output [[1, 4, 27], [4, 25, 216]]
            z = np.power(x, y).astype(np.float32)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_pow_bcast_array')
        _POW()
        _POW_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RandomNormalLike
        self.run_ = nn._RandomNormalLike_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.dtype, self.mean, self.scale, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._OneHot
        self.run_ = nn._OneHot_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.indices_i, self.depth_i, self.values_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _WITHOUT_AXIS():
            on_value = 5
            off_value = 2
            output_type = np.int32
            node = onnx.helper.make_node(
                'OneHot',
                inputs=['indices', 'depth', 'values'],
                outputs=['y']
            )
            indices = np.array([0, 7, 8], dtype=np.int64)
            depth = np.array([12], dtype=np.float32)
            values = np.array([off_value, on_value], dtype=output_type)
            y = one_hot(indices, depth, dtype=output_type)
            y = y * (on_value - off_value) + off_value
            expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_without_axis')
        def _WITH_AXIS():
            axisValue = 1
            on_value = 3
            off_value = 1
            output_type = np.float32
            node = onnx.helper.make_node(
                'OneHot',
                inputs=['indices', 'depth', 'values'],
                outputs=['y'],
                axis=axisValue
            )
            indices = np.array([[1, 9],
                                [2, 4]], dtype=np.float32)
            depth = np.array([10], dtype=np.float32)
            values = np.array([off_value, on_value], dtype=output_type)
            y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
            y = y * (on_value - off_value) + off_value
            expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_axis')
        _WITHOUT_AXIS()
        _WITH_AXIS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RandomUniform
        self.run_ = nn._RandomUniform_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.shape, self.dtype, self.high, self.low, self.seed, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceL1
        self.run_ = nn._ReduceL1_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [2]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceL1',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[3., 7.], [11., 15.], [19., 23.]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [2]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceL1',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_keep_dims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_keep_dims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceL1',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims
            )
            
            data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
            #print(data)
            #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
            
            reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[78.]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_l1_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceLogSum
        self.run_ = nn._ReduceLogSum_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NOKEEPDIMS():
            node = onnx.helper.make_node(
                'ReduceLogSum',
                inputs=['data'],
                outputs=["reduced"],
                axes=[2, 1],
                keepdims=0
            )
            data = np.random.ranf([3, 4, 5]).astype(np.float32)
            reduced = np.log(np.sum(data, axis=(2, 1), keepdims=False))
            expect(node, inputs=[data], outputs=[reduced],
                   name='test_reduce_log_sum_desc_axes')
            
            node = onnx.helper.make_node(
                'ReduceLogSum',
                inputs=['data'],
                outputs=["reduced"],
                axes=[0, 1],
                keepdims=0
            )
            data = np.random.ranf([3, 4, 5]).astype(np.float32)
            reduced = np.log(np.sum(data, axis=(0, 1), keepdims=False))
            expect(node, inputs=[data], outputs=[reduced],
                   name='test_reduce_log_sum_asc_axes')
        def _KEEPDIMS():
            node = onnx.helper.make_node(
                'ReduceLogSum',
                inputs=['data'],
                outputs=["reduced"]
            )
            data = np.random.ranf([3, 4, 5]).astype(np.float32)
            reduced = np.log(np.sum(data, keepdims=True))
            expect(node, inputs=[data], outputs=[reduced],
                   name='test_reduce_log_sum_default')
        _NOKEEPDIMS()
        _KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceLogSumExp
        self.run_ = nn._ReduceLogSumExp_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            node = onnx.helper.make_node(
                'ReduceLogSumExp',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.array(
                [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
                dtype=np.float32)
            reduced = np.log(np.sum(
                np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
            # print(reduced)
            #[[20., 2.31326175]
            # [40.00004578, 2.31326175]
            # [60.00671387, 2.31326175]]
            
            expect(node, inputs=[data], outputs=[reduced],
                  name='test_reduce_log_sum_exp_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.log(np.sum(
                np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
            
            expect(node, inputs=[data], outputs=[reduced],
                name='test_reduce_log_sum_exp_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            node = onnx.helper.make_node(
                'ReduceLogSumExp',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims
            )
            
            data = np.array(
                [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
                dtype=np.float32)
            reduced = np.log(np.sum(np.exp(data),
                                    axis=tuple(axes),
                                    keepdims=keepdims == 1))
            # print(reduced)
            # [[[20., 2.31326175]]
            # [[40.00004578, 2.31326175]]
            # [[60.00671387, 2.31326175]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                  name='test_reduce_log_sum_exp_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.log(np.sum(np.exp(data),
                                    axis=tuple(axes),
                                    keepdims=keepdims == 1))
            
            expect(node, inputs=[data], outputs=[reduced],
                  name='test_reduce_log_sum_exp_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceLogSumExp',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims
            )
            
            data = np.array(
                [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
                dtype=np.float32)
            reduced = np.log(np.sum(np.exp(data),
                                    axis=axes,
                                    keepdims=keepdims == 1))
            # print(reduced)
            # [[[60.00671387]]]
            
            expect(node, inputs=[data], outputs=[reduced],
                  name='test_reduce_log_sum_exp_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.log(np.sum(np.exp(data),
                                    axis=axes,
                                    keepdims=keepdims == 1))
            expect(node, inputs=[data], outputs=[reduced],
                  name='test_reduce_log_sum_exp_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceMax
        self.run_ = nn._ReduceMax_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceMax',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[20., 2.]
            # [40., 2.]
            # [60., 2.]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceMax',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[20., 2.]]
            # [[40., 2.]]
            # [[60., 2.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            node = onnx.helper.make_node(
                'ReduceMax',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            [[[60.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdim_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._OneHotEncoder
        self.run_ = nn._OneHotEncoder_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.cats_int64s, self.cats_strings, self.zeros, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['OneHotEncoder'] = OneHotEncoder





class IsNaN:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._IsNaN
        self.run_ = nn._IsNaN_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ISNAN():
            node = onnx.helper.make_node(
                'IsNaN',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32)
            y = np.isnan(x)
            expect(node, inputs=[x], outputs=[y], name='test_isnan')
        _ISNAN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceMean
        self.run_ = nn._ReduceMean_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[12.5, 1.5]
            # [35., 1.5]
            # [57.5, 1.5]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[12.5, 1.5]]
            # [[35., 1.5]]
            # [[57.5, 1.5]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[18.25]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceMin
        self.run_ = nn._ReduceMin_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceMin',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[5., 1.]
            # [30., 1.]
            # [55., 1.]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceMin', inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[5., 1.]]
            # [[30., 1.]]
            # [[55., 1.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceMin',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
            reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[1.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._TreeEnsembleRegressor
        self.run_ = nn._TreeEnsembleRegressor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.aggregate_function, self.base_values, self.n_targets, self.nodes_falsenodeids, self.nodes_featureids, self.nodes_hitrates, self.nodes_missing_value_tracks_true, self.nodes_modes, self.nodes_nodeids, self.nodes_treeids, self.nodes_truenodeids, self.nodes_values, self.post_transform, self.target_ids, self.target_nodeids, self.target_treeids, self.target_weights, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceProd
        self.run_ = nn._ReduceProd_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[3., 8.]
            # [35., 48.]
            # [99., 120.]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[3., 8.]]
            # [[35., 48.]]
            # [[99., 120.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[4.790016e+08]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceSum
        self.run_ = nn._ReduceSum_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceSum',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[4., 6.]
            # [12., 14.]
            # [20., 22.]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceSum',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[4., 6.]]
            # [[12., 14.]]
            # [[20., 22.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceSum',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(data, axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[78.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(data, axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ReduceSumSquare
        self.run_ = nn._ReduceSumSquare_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.keepdims, self.data_i, self.reduced_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DO_NOT_KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 0
            
            node = onnx.helper.make_node(
                'ReduceSumSquare',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[10., 20.]
            # [74., 100.]
            # [202., 244.]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_do_not_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_do_not_keepdims_random')
        def _KEEPDIMS():
            shape = [3, 2, 2]
            axes = [1]
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceSumSquare',
                inputs=['data'],
                outputs=['reduced'],
                axes=axes,
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
            #print(reduced)
            #[[[10., 20.]]
            # [[74., 100.]]
            # [[202., 244.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_keepdims_random')
        def _DEFAULT_AXES_KEEPDIMS():
            shape = [3, 2, 2]
            axes = None
            keepdims = 1
            
            node = onnx.helper.make_node(
                'ReduceSumSquare',
                inputs=['data'],
                outputs=['reduced'],
                keepdims=keepdims)
            
            data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
            reduced = np.sum(np.square(data), axis=axes, keepdims=keepdims == 1)
            #print(reduced)
            #[[[650.]]]
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_default_axes_keepdims_example')
            
            np.random.seed(0)
            data = np.random.uniform(-10, 10, shape).astype(np.float32)
            reduced = np.sum(np.square(data), axis=axes, keepdims=keepdims == 1)
            
            expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_default_axes_keepdims_random')
        _DO_NOT_KEEPDIMS()
        _KEEPDIMS()
        _DEFAULT_AXES_KEEPDIMS()

layer_map['ReduceSumSquare'] = ReduceSumSquare





class Relu:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Relu
        self.run_ = nn._Relu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _RELU():
            node = onnx.helper.make_node(
                'Relu',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, 0, np.inf)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_relu')
        _RELU()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Reshape
        self.run_ = nn._Reshape_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.data_i, self.shape_i, self.reshaped_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _RESHAPE():
            original_shape = [2, 3, 4]
            test_cases = {
                'reordered_dims': np.array([4, 2, 3], dtype=np.int64),
                'reduced_dims': np.array([3, 8], dtype=np.int64),
                'extended_dims': np.array([3, 2, 2, 2], dtype=np.int64),
                'one_dim': np.array([24], dtype=np.int64),
                'negative_dim': np.array([6, -1, 2], dtype=np.int64),
            }
            data = np.random.random_sample(original_shape).astype(np.float32)
            
            for test_name, shape in test_cases.items():
                node = onnx.helper.make_node(
                    'Reshape',
                    inputs=['data', 'shape'],
                    outputs=['reshaped'],
                )
            
                reshaped = np.reshape(data, shape)
                expect(node, inputs=[data, shape], outputs=[reshaped],
                       name='test_reshape_' + test_name)
        _RESHAPE()

layer_map['Reshape'] = Reshape





class Shape:
    name = None
    data_i = str()
    shape_o = str()

    #parameters

    input_params = ["data_i"]
    output_params = ["shape_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Shape
        self.run_ = nn._Shape_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.data_i, self.shape_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SHAPE():
            node = onnx.helper.make_node(
                'Shape',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).astype(np.float32)
            y = np.array([
                2, 3,
            ]).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_shape_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.array(x.shape).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_shape')
        _SHAPE()

layer_map['Shape'] = Shape





class Sigmoid:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sigmoid
        self.run_ = nn._Sigmoid_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SIGMOID():
            node = onnx.helper.make_node(
                'Sigmoid',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = 1.0 / (1.0 + np.exp(np.negative(x)))  # expected output [0.26894143, 0.5, 0.7310586]
            expect(node, inputs=[x], outputs=[y],
                   name='test_sigmoid_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = 1.0 / (1.0 + np.exp(np.negative(x)))
            expect(node, inputs=[x], outputs=[y],
                   name='test_sigmoid')
        _SIGMOID()

layer_map['Sigmoid'] = Sigmoid





class Size:
    name = None
    data_i = str()
    size_o = str()

    #parameters

    input_params = ["data_i"]
    output_params = ["size_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Size
        self.run_ = nn._Size_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.data_i, self.size_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SIZE():
            node = onnx.helper.make_node(
                'Size',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([
                [1, 2, 3],
                [4, 5, 6],
            ]).astype(np.float32)
            y = np.array(6).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_size_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.array(x.size).astype(np.int64)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_size')
        _SIZE()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Softmax
        self.run_ = nn._Softmax_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SOFTMAX():
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
            )
            x = np.array([[-1, 0, 1]]).astype(np.float32)
            # expected output [[0.09003058, 0.24472848, 0.66524094]]
            y = np.exp(x) / np.sum(np.exp(x), axis=1)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_example')
        def _SOFTMAX_AXIS():
            def softmax_2d(x):  # type: (np.ndarray) -> np.ndarray
                max_x = np.max(x, axis=1).reshape((-1, 1))
                exp_x = np.exp(x - max_x)
                return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
            
            x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
            # expected output [[0.0320586, 0.08714432, 0.23688284, 0.64391428],
            #                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]
            y = softmax_2d(x)
            
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
            )
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_large_number')
            
            x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
                axis=0,
            )
            y = softmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_axis_0')
            
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
                axis=1,
            )
            y = softmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_axis_1')
            
            # default axis is 1
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
            )
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_default_axis')
            
            node = onnx.helper.make_node(
                'Softmax',
                inputs=['x'],
                outputs=['y'],
                axis=2,
            )
            y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softmax_axis_2')
        _SOFTMAX()
        _SOFTMAX_AXIS()

layer_map['Softmax'] = Softmax





class Softplus:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Softplus
        self.run_ = nn._Softplus_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SOFTPLUS():
            node = onnx.helper.make_node(
                'Softplus',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.log(np.exp(x) + 1)  # expected output [0.31326166, 0.69314718, 1.31326163]
            expect(node, inputs=[x], outputs=[y],
                   name='test_softplus_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.log(np.exp(x) + 1)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softplus')
        _SOFTPLUS()

layer_map['Softplus'] = Softplus





class Softsign:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Softsign
        self.run_ = nn._Softsign_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SOFTSIGN():
            node = onnx.helper.make_node(
                'Softsign',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.array([-0.5, 0, 0.5]).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_softsign_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = x / (1 + np.abs(x))
            expect(node, inputs=[x], outputs=[y],
                   name='test_softsign')
        _SOFTSIGN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._SpaceToDepth
        self.run_ = nn._SpaceToDepth_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.blocksize, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._TfIdfVectorizer
        self.run_ = nn._TfIdfVectorizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.max_gram_length, self.max_skip_count, self.min_gram_length, self.mode, self.ngram_counts, self.ngram_indexes, self.pool_int64s, self.pool_strings, self.weights, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TF_ONLY_BIGRAMS_SKIP0():
            input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
            output = np.array([0., 0., 0., 0., 1., 1., 1.]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=2,
                max_gram_length=2,
                max_skip_count=0,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_only_bigrams_skip0')
        def _TF_BATCH_ONLYBIGRAMS_SKIP0():
            input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
            output = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 1.]]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=2,
                max_gram_length=2,
                max_skip_count=0,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_onlybigrams_skip0')
        def _TF_ONLYBIGRAMS_LEVELEMPTY():
            input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
            output = np.array([1., 1., 1.]).astype(np.float32)
            
            ngram_counts = np.array([0, 0]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2]).astype(np.int64)
            pool_int64s = np.array([    # unigrams none
                                   5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=2,
                max_gram_length=2,
                max_skip_count=0,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_onlybigrams_levelempty')
        def _TF_ONLYBIGRAMS_SKIP5():
            input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
            output = np.array([0., 0., 0., 0., 1., 3., 1.]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=2,
                max_gram_length=2,
                max_skip_count=5,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_onlybigrams_skip5')
        def _TF_BATCH_ONLYBIGRAMS_SKIP5():
            input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
            output = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 1., 1.]]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=2,
                max_gram_length=2,
                max_skip_count=5,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_onlybigrams_skip5')
        def _TF_UNIANDBIGRAMS_SKIP5():
            input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
            output = np.array([0., 3., 1., 0., 1., 3., 1.]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=1,
                max_gram_length=2,
                max_skip_count=5,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_uniandbigrams_skip5')
        def _TF_BATCH_UNIANDBIGRAMS_SKIP5():
            input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
            output = np.array([[0., 3., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 1., 1.]]).astype(np.float32)
            
            ngram_counts = np.array([0, 4]).astype(np.int64)
            ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
            pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                                    5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams
            
            helper = TfIdfVectorizerHelper(
                mode='TF',
                min_gram_length=1,
                max_gram_length=2,
                max_skip_count=5,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s
            )
            node = helper.make_node_noweights()
            expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_uniandbigrams_skip5')
        _TF_ONLY_BIGRAMS_SKIP0()
        _TF_BATCH_ONLYBIGRAMS_SKIP0()
        _TF_ONLYBIGRAMS_LEVELEMPTY()
        _TF_ONLYBIGRAMS_SKIP5()
        _TF_BATCH_ONLYBIGRAMS_SKIP5()
        _TF_UNIANDBIGRAMS_SKIP5()
        _TF_BATCH_UNIANDBIGRAMS_SKIP5()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Split
        self.run_ = nn._Split_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.split, self.input_i)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _1D():
            input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
            
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2', 'output_3'],
                axis=0
            )
            
            expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_1d')
            
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2'],
                axis=0,
                split=[2, 4]
            )
            
            expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4., 5., 6.]).astype(np.float32)]
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_variable_parts_1d')
        def _2D():
            input = np.array([[1., 2., 3., 4., 5., 6.],
                              [7., 8., 9., 10., 11., 12.]]).astype(np.float32)
            
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2'],
                axis=1
            )
            
            expected_outputs = [np.array([[1., 2., 3.], [7., 8., 9.]]).astype(np.float32),
                                np.array([[4., 5., 6.], [10., 11., 12.]]).astype(np.float32)]
            
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_2d')
            
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2'],
                axis=1,
                split=[2, 4]
            )
            
            expected_outputs = [np.array([[1., 2.], [7., 8.]]).astype(np.float32),
                                np.array([[3., 4., 5., 6.], [9., 10., 11., 12.]]).astype(np.float32)]
            
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_variable_parts_2d')
        def _DEFAULT_VALUES():
            input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
            
            # If axis is not specified, split is applied on default axis 0
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2', 'output_3']
            )
            
            expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_default_axis')
            
            node = onnx.helper.make_node(
                'Split',
                inputs=['input'],
                outputs=['output_1', 'output_2'],
                split=[2, 4]
            )
            
            expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4., 5., 6.]).astype(np.float32)]
            expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_variable_parts_default_axis')
        _1D()
        _2D()
        _DEFAULT_VALUES()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Imputer
        self.run_ = nn._Imputer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.imputed_value_floats, self.imputed_value_int64s, self.replaced_value_float, self.replaced_value_int64, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['Imputer'] = Imputer





class Sqrt:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sqrt
        self.run_ = nn._Sqrt_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SQRT():
            node = onnx.helper.make_node(
                'Sqrt',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([1, 4, 9]).astype(np.float32)
            y = np.sqrt(x)  # expected output [1., 2., 3.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_sqrt_example')
            
            x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
            y = np.sqrt(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_sqrt')
        _SQRT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Squeeze
        self.run_ = nn._Squeeze_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.data_i, self.squeezed_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SQUEEZE():
            node = onnx.helper.make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
                axes=[0],
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.float32)
            y = np.squeeze(x, axis=0)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_squeeze')
        _SQUEEZE()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._TopK
        self.run_ = nn._TopK_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.X_i, self.K_i, self.Values_o, self.Indices_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TOP_K():
            node = onnx.helper.make_node(
                'TopK',
                inputs=['x', 'k'],
                outputs=['values', 'indices'],
            )
            X = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ], dtype=np.float32)
            K = np.array([3], dtype=np.int64)
            values_ref = np.array([
                [3, 2, 1],
                [7, 6, 5],
                [11, 10, 9],
            ], dtype=np.float32)
            indices_ref = np.array([
                [3, 2, 1],
                [3, 2, 1],
                [3, 2, 1],
            ], dtype=np.int64)
            
            expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
                   name='test_top_k')
        _TOP_K()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sub
        self.run_ = nn._Sub_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SUB():
            node = onnx.helper.make_node(
                'Sub',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([1, 2, 3]).astype(np.float32)
            y = np.array([3, 2, 1]).astype(np.float32)
            z = x - y  # expected output [-2., 0., 2.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_sub_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(3, 4, 5).astype(np.float32)
            z = x - y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_sub')
        def _SUB_BROADCAST():
            node = onnx.helper.make_node(
                'Sub',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.random.randn(5).astype(np.float32)
            z = x - y
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_sub_bcast')
        _SUB()
        _SUB_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sum
        self.run_ = nn._Sum_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.sum_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SUM():
            data_0 = np.array([3, 0, 2]).astype(np.float32)
            data_1 = np.array([1, 3, 4]).astype(np.float32)
            data_2 = np.array([2, 6, 6]).astype(np.float32)
            result = np.array([6, 9, 12]).astype(np.float32)
            node = onnx.helper.make_node(
                'Sum',
                inputs=['data_0', 'data_1', 'data_2'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
                   name='test_sum_example')
            
            node = onnx.helper.make_node(
                'Sum',
                inputs=['data_0'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0], outputs=[data_0],
                   name='test_sum_one_input')
            
            result = np.add(data_0, data_1)
            node = onnx.helper.make_node(
                'Sum',
                inputs=['data_0', 'data_1'],
                outputs=['result'],
            )
            expect(node, inputs=[data_0, data_1], outputs=[result],
                   name='test_sum_two_inputs')
        _SUM()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Shrink
        self.run_ = nn._Shrink_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.bias, self.lambd, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _HARD_SHRINK():
            node = onnx.helper.make_node(
                'Shrink',
                inputs=['x'],
                outputs=['y'],
                lambd=1.5,
            )
            X = np.arange(-2.0, 2.1, dtype=np.float32)
            Y = np.array([-2, 0, 0, 0, 2], dtype=np.float32)
            expect(node, inputs=[X], outputs=[Y],
                   name='test_shrink_hard')
        def _SOFT_SHRINK():
            node = onnx.helper.make_node(
                'Shrink',
                inputs=['x'],
                outputs=['y'],
                lambd=1.5,
                bias=1.5,
            )
            X = np.arange(-2.0, 2.1, dtype=np.float32)
            Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
            expect(node, inputs=[X], outputs=[Y],
                   name='test_shrink_soft')
        _HARD_SHRINK()
        _SOFT_SHRINK()

layer_map['Shrink'] = Shrink





class Tanh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Tanh
        self.run_ = nn._Tanh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TANH():
            node = onnx.helper.make_node(
                'Tanh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
            expect(node, inputs=[x], outputs=[y],
                   name='test_tanh_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.tanh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_tanh')
        _TANH()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Transpose
        self.run_ = nn._Transpose_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.perm, self.data_i, self.transposed_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEFAULT():
            shape = (2, 3, 4)
            data = np.random.random_sample(shape).astype(np.float32)
            
            node = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed']
            )
            
            transposed = np.transpose(data)
            expect(node, inputs=[data], outputs=[transposed],
                   name='test_transpose_default')
        def _ALL_PERMUTATIONS():
            shape = (2, 3, 4)
            data = np.random.random_sample(shape).astype(np.float32)
            permutations = list(itertools.permutations(np.arange(len(shape))))
            
            for i in range(len(permutations)):
                node = onnx.helper.make_node(
                    'Transpose',
                    inputs=['data'],
                    outputs=['transposed'],
                    perm=permutations[i]
                )
                transposed = np.transpose(data, permutations[i])
                expect(node, inputs=[data], outputs=[transposed],
                       name='test_transpose_all_permutations_' + str(i))
        _DEFAULT()
        _ALL_PERMUTATIONS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Unsqueeze
        self.run_ = nn._Unsqueeze_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.data_i, self.expanded_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _UNSQUEEZE():
            node = onnx.helper.make_node(
                'Unsqueeze',
                inputs=['x'],
                outputs=['y'],
                axes=[0],
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.expand_dims(x, axis=0)
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_unsqueeze')
        _UNSQUEEZE()

layer_map['Unsqueeze'] = Unsqueeze





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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._SVMClassifier
        self.run_ = nn._SVMClassifier_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.classlabels_ints, self.classlabels_strings, self.coefficients, self.kernel_params, self.kernel_type, self.post_transform, self.prob_a, self.prob_b, self.rho, self.support_vectors, self.vectors_per_class, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Xor
        self.run_ = nn._Xor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _XOR():
            node = onnx.helper.make_node(
                'Xor',
                inputs=['x', 'y'],
                outputs=['xor'],
            )
            
            # 2d
            x = (np.random.randn(3, 4) > 0).astype(np.bool)
            y = (np.random.randn(3, 4) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor2d')
            
            # 3d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor3d')
            
            # 4d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor4d')
        def _XOR_BROADCAST():
            node = onnx.helper.make_node(
                'Xor',
                inputs=['x', 'y'],
                outputs=['xor'],
            )
            
            # 3d vs 1d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(5) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor_bcast3v1d')
            
            # 3d vs 2d
            x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
            y = (np.random.randn(4, 5) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor_bcast3v2d')
            
            # 4d vs 2d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(5, 6) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor_bcast4v2d')
            
            # 4d vs 3d
            x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
            y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor_bcast4v3d')
            
            # 4d vs 4d
            x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
            y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
            z = np.logical_xor(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_xor_bcast4v4d')
        _XOR()
        _XOR_BROADCAST()

layer_map['Xor'] = Xor





class Acos:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Acos
        self.run_ = nn._Acos_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ACOS():
            node = onnx.helper.make_node(
                'Acos',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-0.5, 0, 0.5]).astype(np.float32)
            y = np.arccos(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_acos_example')
            
            x = np.random.rand(3, 4, 5).astype(np.float32)
            y = np.arccos(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_acos')
        _ACOS()

layer_map['Acos'] = Acos





class Asin:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Asin
        self.run_ = nn._Asin_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ASIN():
            node = onnx.helper.make_node(
                'Asin',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-0.5, 0, 0.5]).astype(np.float32)
            y = np.arcsin(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_asin_example')
            
            x = np.random.rand(3, 4, 5).astype(np.float32)
            y = np.arcsin(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_asin')
        _ASIN()

layer_map['Asin'] = Asin





class Atan:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Atan
        self.run_ = nn._Atan_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ATAN():
            node = onnx.helper.make_node(
                'Atan',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.arctan(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_atan_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.arctan(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_atan')
        _ATAN()

layer_map['Atan'] = Atan





class Cos:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Cos
        self.run_ = nn._Cos_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _COS():
            node = onnx.helper.make_node(
                'Cos',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.cos(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_cos_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.cos(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_cos')
        _COS()

layer_map['Cos'] = Cos





class Sin:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sin
        self.run_ = nn._Sin_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SIN():
            node = onnx.helper.make_node(
                'Sin',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.sin(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_sin_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.sin(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_sin')
        _SIN()

layer_map['Sin'] = Sin





class Tan:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Tan
        self.run_ = nn._Tan_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _TAN():
            node = onnx.helper.make_node(
                'Tan',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.tan(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_tan_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.tan(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_tan')
        _TAN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Multinomial
        self.run_ = nn._Multinomial_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.dtype, self.sample_size, self.seed, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Scan
        self.run_ = nn._Scan_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.body, self.num_scan_inputs, self.scan_input_axes, self.scan_input_directions, self.scan_output_axes, self.scan_output_directions, self.x0_i, self.x1_i, self.x2_i, self.x3_i, self.x4_i, self.x5_i, self.x6_i, self.x7_i, self.x8_i, self.x9_i, self.x10_i, self.x11_i, self.x12_i, self.x13_i, self.x14_i, self.x15_i, self.x16_i, self.x17_i, self.x18_i, self.x19_i, self.x20_i, self.x21_i, self.x22_i, self.x23_i, self.x24_i, self.x25_i, self.x26_i, self.x27_i, self.x28_i, self.x29_i, self.x30_i, self.x31_i, self.y0_o, self.y1_o, self.y2_o, self.y3_o, self.y4_o, self.y5_o, self.y6_o, self.y7_o, self.y8_o, self.y9_o, self.y10_o, self.y11_o, self.y12_o, self.y13_o, self.y14_o, self.y15_o, self.y16_o, self.y17_o, self.y18_o, self.y19_o, self.y20_o, self.y21_o, self.y22_o, self.y23_o, self.y24_o, self.y25_o, self.y26_o, self.y27_o, self.y28_o, self.y29_o, self.y30_o, self.y31_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SCAN_8():
            # Given an input sequence [x1, ..., xN], sum up its elements using a scan
            # returning the final state (x1+x2+...+xN) as well the scan_output
            # [x1, x1+x2, ..., x1+x2+...+xN]
            #
            # create graph to represent scan body
            sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
            next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
            sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
            scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
            add_node = onnx.helper.make_node(
                'Add',
                inputs=['sum_in', 'next'],
                outputs=['sum_out']
            )
            id_node = onnx.helper.make_node(
                'Identity',
                inputs=['sum_out'],
                outputs=['scan_out']
            )
            scan_body = onnx.helper.make_graph(
                [add_node, id_node],
                'scan_body',
                [sum_in, next],
                [sum_out, scan_out]
            )
            # create scan op node
            no_sequence_lens = ''   # optional input, not supplied
            node = onnx.helper.make_node(
                'Scan',
                inputs=[no_sequence_lens, 'initial', 'x'],
                outputs=['y', 'z'],
                num_scan_inputs=1,
                body=scan_body
            )
            # create inputs for batch-size 1, sequence-length 3, inner dimension 2
            initial = np.array([0, 0]).astype(np.float32).reshape((1, 2))
            x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((1, 3, 2))
            # final state computed = [1 + 3 + 5, 2 + 4 + 6]
            y = np.array([9, 12]).astype(np.float32).reshape((1, 2))
            # scan-output computed
            z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((1, 3, 2))
            
            expect(node, inputs=[initial, x], outputs=[y, z],
                   name='test_scan_sum', opset_imports=[onnx.helper.make_opsetid("", 8)])
        def _SCAN_9():
            # Given an input sequence [x1, ..., xN], sum up its elements using a scan
            # returning the final state (x1+x2+...+xN) as well the scan_output
            # [x1, x1+x2, ..., x1+x2+...+xN]
            #
            # create graph to represent scan body
            sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
            next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
            sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
            scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
            add_node = onnx.helper.make_node(
                'Add',
                inputs=['sum_in', 'next'],
                outputs=['sum_out']
            )
            id_node = onnx.helper.make_node(
                'Identity',
                inputs=['sum_out'],
                outputs=['scan_out']
            )
            scan_body = onnx.helper.make_graph(
                [add_node, id_node],
                'scan_body',
                [sum_in, next],
                [sum_out, scan_out]
            )
            # create scan op node
            node = onnx.helper.make_node(
                'Scan',
                inputs=['initial', 'x'],
                outputs=['y', 'z'],
                num_scan_inputs=1,
                body=scan_body
            )
            # create inputs for sequence-length 3, inner dimension 2
            initial = np.array([0, 0]).astype(np.float32).reshape((2,))
            x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))
            # final state computed = [1 + 3 + 5, 2 + 4 + 6]
            y = np.array([9, 12]).astype(np.float32).reshape((2,))
            # scan-output computed
            z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))
            
            expect(node, inputs=[initial, x], outputs=[y, z],
                   name='test_scan9_sum', opset_imports=[onnx.helper.make_opsetid("", 9)])
        _SCAN_8()
        _SCAN_9()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Compress
        self.run_ = nn._Compress_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.input_i, self.condition_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _COMPRESS_0():
            node = onnx.helper.make_node(
                'Compress',
                inputs=['input', 'condition'],
                outputs=['output'],
                axis=0,
            )
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
            condition = np.array([0, 1, 1])
            output = np.compress(condition, input, axis=0)
            #print(output)
            #[[ 3.  4.]
            # [ 5.  6.]]
            
            expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
                   name='test_compress_0')
        def _COMPRESS_1():
            node = onnx.helper.make_node(
                'Compress',
                inputs=['input', 'condition'],
                outputs=['output'],
                axis=1,
            )
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
            condition = np.array([0, 1])
            output = np.compress(condition, input, axis=1)
            #print(output)
            #[[ 2.]
            # [ 4.]
            # [ 6.]]
            
            expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
                   name='test_compress_1')
        def _COMPRESS_DEFAULT_AXIS():
            node = onnx.helper.make_node(
                'Compress',
                inputs=['input', 'condition'],
                outputs=['output'],
            )
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
            condition = np.array([0, 1, 0, 0, 1])
            output = np.compress(condition, input)
            #print(output)
            #[ 2., 5.]
            
            expect(node, inputs=[input, condition.astype(np.bool)], outputs=[output],
                   name='test_compress_default_axis')
        _COMPRESS_0()
        _COMPRESS_1()
        _COMPRESS_DEFAULT_AXIS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ConstantOfShape
        self.run_ = nn._ConstantOfShape_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.value, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _FLOAT_ONES():
            x = np.array([4, 3, 2])
            tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                                   [1], [1])
            node = onnx.helper.make_node(
                'ConstantOfShape',
                inputs=['x'],
                outputs=['y'],
                value=tensor_value,
            )
            
            y = np.ones(x, dtype=np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_constantofshape_float_ones')
        def _INT32_ZEROS():
            x = np.array([10, 6])
            tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                                   [1], [0])
            node = onnx.helper.make_node(
                'ConstantOfShape',
                inputs=['x'],
                outputs=['y'],
                value=tensor_value,
            )
            y = np.zeros(x, dtype=np.int32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_constantofshape_int_zeros')
        _FLOAT_ONES()
        _INT32_ZEROS()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MaxUnpool
        self.run_ = nn._MaxUnpool_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.kernel_shape, self.pads, self.strides, self.X_i, self.I_i, self.output_shape_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _WITHOUT_OUTPUT_SHAPE():
            node = onnx.helper.make_node(
                'MaxUnpool',
                inputs=['xT', 'xI'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            )
            xT = np.array([[[[1, 2],
                             [3, 4]]]], dtype=np.float32)
            xI = np.array([[[[5, 7],
                             [13, 15]]]], dtype=np.int64)
            y = np.array([[[[0, 0, 0, 0],
                            [0, 1, 0, 2],
                            [0, 0, 0, 0],
                            [0, 3, 0, 4]]]], dtype=np.float32)
            expect(node, inputs=[xT, xI], outputs=[y], name='test_maxunpool_export_without_output_shape')
        def _WITH_OUTPUT_SHAPE():
            node = onnx.helper.make_node(
                'MaxUnpool',
                inputs=['xT', 'xI', 'output_shape'],
                outputs=['y'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            )
            xT = np.array([[[[5, 6],
                             [7, 8]]]], dtype=np.float32)
            xI = np.array([[[[5, 7],
                             [13, 15]]]], dtype=np.int64)
            output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
            y = np.array([[[[0, 0, 0, 0, 0],
                            [0, 5, 0, 6, 0],
                            [0, 0, 0, 0, 0],
                            [0, 7, 0, 8, 0],
                            [0, 0, 0, 0, 0]]]], dtype=np.float32)
            expect(node, inputs=[xT, xI, output_shape], outputs=[y], name='test_maxunpool_export_with_output_shape')
        _WITHOUT_OUTPUT_SHAPE()
        _WITH_OUTPUT_SHAPE()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Scatter
        self.run_ = nn._Scatter_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axis, self.data_i, self.indices_i, self.updates_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SCATTER_WITHOUT_AXIS():
            node = onnx.helper.make_node(
                'Scatter',
                inputs=['data', 'indices', 'updates'],
                outputs=['y'],
            )
            data = np.zeros((3, 3), dtype=np.float32)
            indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
            updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)
            
            y = np.array([
                [2.0, 1.1, 0.0],
                [1.0, 0.0, 2.2],
                [0.0, 2.1, 1.2]
            ], dtype=np.float32)
            
            expect(node, inputs=[data, indices, updates], outputs=[y],
                   name='test_scatter_without_axis')
        def _SCATTER_WITH_AXIS():
            node = onnx.helper.make_node(
                'Scatter',
                inputs=['data', 'indices', 'updates'],
                outputs=['y'],
                axis=1,
            )
            data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
            indices = np.array([[1, 3]], dtype=np.int64)
            updates = np.array([[1.1, 2.1]], dtype=np.float32)
            
            y = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)
            
            expect(node, inputs=[data, indices, updates], outputs=[y],
                   name='test_scatter_with_axis')
        _SCATTER_WITHOUT_AXIS()
        _SCATTER_WITH_AXIS()

layer_map['Scatter'] = Scatter





class Sinh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sinh
        self.run_ = nn._Sinh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SINH():
            node = onnx.helper.make_node(
                'Sinh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.sinh(x)  # expected output [-1.17520118,  0.,  1.17520118]
            expect(node, inputs=[x], outputs=[y],
                   name='test_sinh_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.sinh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_sinh')
        _SINH()

layer_map['Sinh'] = Sinh





class Cosh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Cosh
        self.run_ = nn._Cosh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _COSH():
            node = onnx.helper.make_node(
                'Cosh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.cosh(x)  # expected output [1.54308069,  1.,  1.54308069]
            expect(node, inputs=[x], outputs=[y],
                   name='test_cosh_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.cosh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_cosh')
        _COSH()

layer_map['Cosh'] = Cosh





class Asinh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Asinh
        self.run_ = nn._Asinh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ASINH():
            node = onnx.helper.make_node(
                'Asinh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-1, 0, 1]).astype(np.float32)
            y = np.arcsinh(x)  # expected output [-0.88137358,  0.,  0.88137358]
            expect(node, inputs=[x], outputs=[y],
                   name='test_asinh_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.arcsinh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_asinh')
        _ASINH()

layer_map['Asinh'] = Asinh





class Acosh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Acosh
        self.run_ = nn._Acosh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ACOSH():
            node = onnx.helper.make_node(
                'Acosh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([10, np.e, 1]).astype(np.float32)
            y = np.arccosh(x)  # expected output [2.99322295,  1.65745449,  0.]
            expect(node, inputs=[x], outputs=[y],
                   name='test_acosh_example')
            
            x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
            y = np.arccosh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_acosh')
        _ACOSH()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._NonMaxSuppression
        self.run_ = nn._NonMaxSuppression_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.center_point_box, self.boxes_i, self.scores_i, self.max_output_boxes_per_class_i, self.iou_threshold_i, self.score_threshold_i, self.selected_indices_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NONMAXSUPPRESSION_SUPPRESS_BY_IOU():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU')
        def _NONMAXSUPPRESSION_SUPPRESS_BY_IOU_AND_SCORES():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.4]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU_and_scores')
        def _NONMAXSUPPRESSION_FLIPPED_COORDINATES():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.9, 1.0, -0.1],
                [0.0, 10.0, 1.0, 11.0],
                [1.0, 10.1, 0.0, 11.1],
                [1.0, 101.0, 0.0, 100.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_flipped_coordinates')
        def _NONMAXSUPPRESSION_LIMIT_OUTPUT_SIZE():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([2]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_limit_output_size')
        def _NONMAXSUPPRESSION_SINGLE_BOX():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')
        def _NONMAXSUPPRESSION_IDENTICAL_BOXES():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_identical_boxes')
        def _NONMAXSUPPRESSION_CENTER_POINT_BOX_FORMAT():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices'],
                center_point_box=1
            )
            boxes = np.array([[
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.6, 1.0, 1.0],
                [0.5, 0.4, 1.0, 1.0],
                [0.5, 10.5, 1.0, 1.0],
                [0.5, 10.6, 1.0, 1.0],
                [0.5, 100.5, 1.0, 1.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([3]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_center_point_box_format')
        def _NONMAXSUPPRESSION_TWO_CLASSES():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0]
            ]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                                [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([2]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_classes')
        def _NONMAXSUPPRESSION_TWO_BATCHES():
            node = onnx.helper.make_node(
                'NonMaxSuppression',
                inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
                outputs=['selected_indices']
            )
            boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                               [0.0, 0.1, 1.0, 1.1],
                               [0.0, -0.1, 1.0, 0.9],
                               [0.0, 10.0, 1.0, 11.0],
                               [0.0, 10.1, 1.0, 11.1],
                               [0.0, 100.0, 1.0, 101.0]],
                              [[0.0, 0.0, 1.0, 1.0],
                               [0.0, 0.1, 1.0, 1.1],
                               [0.0, -0.1, 1.0, 0.9],
                               [0.0, 10.0, 1.0, 11.0],
                               [0.0, 10.1, 1.0, 11.1],
                               [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                               [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([2]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            selected_indices = np.array([[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)
            
            expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_batches')
        _NONMAXSUPPRESSION_SUPPRESS_BY_IOU()
        _NONMAXSUPPRESSION_SUPPRESS_BY_IOU_AND_SCORES()
        _NONMAXSUPPRESSION_FLIPPED_COORDINATES()
        _NONMAXSUPPRESSION_LIMIT_OUTPUT_SIZE()
        _NONMAXSUPPRESSION_SINGLE_BOX()
        _NONMAXSUPPRESSION_IDENTICAL_BOXES()
        _NONMAXSUPPRESSION_CENTER_POINT_BOX_FORMAT()
        _NONMAXSUPPRESSION_TWO_CLASSES()
        _NONMAXSUPPRESSION_TWO_BATCHES()

layer_map['NonMaxSuppression'] = NonMaxSuppression





class Atanh:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Atanh
        self.run_ = nn._Atanh_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ATANH():
            node = onnx.helper.make_node(
                'Atanh',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array([-0.5, 0, 0.5]).astype(np.float32)
            y = np.arctanh(x)  # expected output [-0.54930615,  0.,  0.54930615]
            expect(node, inputs=[x], outputs=[y],
                   name='test_atanh_example')
            
            x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
            y = np.arctanh(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_atanh')
        _ATANH()

layer_map['Atanh'] = Atanh





class Sign:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Sign
        self.run_ = nn._Sign_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _SIGN():
            node = onnx.helper.make_node(
                'Sign',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.array(range(-5, 6)).astype(np.float32)
            y = np.sign(x)
            expect(node, inputs=[x], outputs=[y],
                   name='test_sign')
        _SIGN()

layer_map['Sign'] = Sign





class Erf:
    name = None
    input_i = str()
    output_o = str()

    #parameters

    input_params = ["input_i"]
    output_params = ["output_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Erf
        self.run_ = nn._Erf_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.input_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ERF():
            node = onnx.helper.make_node(
                'Erf',
                inputs=['x'],
                outputs=['y'],
            )
            
            x = np.random.randn(1, 3, 32, 32).astype(np.float32)
            y = np.vectorize(math.erf)(x).astype(np.float32)
            expect(node, inputs=[x], outputs=[y],
                   name='test_erf')
        _ERF()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Where
        self.run_ = nn._Where_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.condition_i, self.X_i, self.Y_i, self.output_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _WHERE():
            node = onnx.helper.make_node(
                'Where',
                inputs=['condition', 'x', 'y'],
                outputs=['z'],
            )
            
            condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
            x = np.array([[1, 2], [3, 4]], dtype=np.float32)
            y = np.array([[9, 8], [7, 6]], dtype=np.float32)
            z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
            expect(node, inputs=[condition, x, y], outputs=[z],
                   name='test_where_example')
        _WHERE()

layer_map['Where'] = Where





class NonZero:
    name = None
    X_i = str()
    Y_o = str()

    #parameters

    input_params = ["X_i"]
    output_params = ["Y_o"]
    attribute_params = []
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._NonZero
        self.run_ = nn._NonZero_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NONZERO():
            node = onnx.helper.make_node(
                'NonZero',
                inputs=['condition'],
                outputs=['result'],
            )
            
            condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
            result = np.array((np.nonzero(condition)))  # expected output [[0, 1, 1], [0, 0, 1]]
            expect(node, inputs=[condition], outputs=[result],
                   name='test_nonzero_example')
        _NONZERO()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MeanVarianceNormalization
        self.run_ = nn._MeanVarianceNormalization_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.axes, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MEANVARIANCENORMALIZATION():
            node = onnx.helper.make_node(
                'MeanVarianceNormalization',
                inputs=['X'],
                outputs=['Y']
            )
            
            input_data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
                [[0.02916367], [0.12964272], [0.5060197]],
                [[0.79538304], [0.9411346], [0.9546573]]],
                [[[0.17730942], [0.46192095], [0.26480448]],
                [[0.6746842], [0.01665257], [0.62473077]],
                [[0.9240844], [0.9722341], [0.11965699]]],
                [[[0.41356155], [0.9129373], [0.59330076]],
                [[0.81929934], [0.7862604], [0.11799799]],
                [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)
            
            # Calculate expected output data
            data_mean = np.mean(input_data, axis=(0, 2, 3), keepdims=1)
            data_mean_squared = np.power(data_mean, 2)
            data_squared = np.power(input_data, 2)
            data_squared_mean = np.mean(data_squared, axis=(0, 2, 3), keepdims=1)
            std = np.sqrt(data_squared_mean - data_mean_squared)
            expected_output = (input_data - data_mean) / (std + 1e-9)
            
            expect(node, inputs=[input_data], outputs=[expected_output],
                   name='test_mvn')
        _MEANVARIANCENORMALIZATION()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._StringNormalizer
        self.run_ = nn._StringNormalizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.case_change_action, self.is_case_sensitive, self.locale, self.stopwords, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _NOSTOPWORDS_NOCHANGECASE():
            input = np.array([u'monday', u'tuesday']).astype(np.object)
            output = input
            
            # No stopwords. This is a NOOP
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                is_case_sensitive=1,
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_nostopwords_nochangecase')
        def _MONDAY_CASESENSINTIVE_NOCHANGECASE():
            input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
            output = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)
            stopwords = [u'monday']
            
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                is_case_sensitive=1,
                stopwords=stopwords
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_nochangecase')
        def _MONDAY_CASESENSINTIVE_LOWER():
            input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
            output = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)
            stopwords = [u'monday']
            
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                case_change_action='LOWER',
                is_case_sensitive=1,
                stopwords=stopwords
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_lower')
        def _MONDAY_CASESENSINTIVE_UPPER():
            input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
            output = np.array([u'TUESDAY', u'WEDNESDAY', u'THURSDAY']).astype(np.object)
            stopwords = [u'monday']
            
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                case_change_action='UPPER',
                is_case_sensitive=1,
                stopwords=stopwords
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper')
        def _MONDAY_EMPTY_OUTPUT():
            input = np.array([u'monday', u'monday']).astype(np.object)
            output = np.array([u'']).astype(np.object)
            stopwords = [u'monday']
            
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                case_change_action='UPPER',
                is_case_sensitive=1,
                stopwords=stopwords
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')
        def _MONDAY_INSENSINTIVE_UPPER_TWODIM():
            input = np.array([u'Monday', u'tuesday', u'wednesday', u'Monday', u'tuesday', u'wednesday']).astype(np.object).reshape([1, 6])
            
            # It does upper case cecedille, accented E
            # and german umlaut but fails
            # with german eszett
            output = np.array([u'TUESDAY', u'WEDNESDAY', u'TUESDAY', u'WEDNESDAY']).astype(np.object).reshape([1, 4])
            stopwords = [u'monday']
            
            node = onnx.helper.make_node(
                'StringNormalizer',
                inputs=['x'],
                outputs=['y'],
                case_change_action='UPPER',
                stopwords=stopwords
            )
            expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')
        _NOSTOPWORDS_NOCHANGECASE()
        _MONDAY_CASESENSINTIVE_NOCHANGECASE()
        _MONDAY_CASESENSINTIVE_LOWER()
        _MONDAY_CASESENSINTIVE_UPPER()
        _MONDAY_EMPTY_OUTPUT()
        _MONDAY_INSENSINTIVE_UPPER_TWODIM()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Mod
        self.run_ = nn._Mod_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.fmod, self.A_i, self.B_i, self.C_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _FLOAT_MIXED_SIGN():
            node = onnx.helper.make_node(
                'Mod',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
            y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])
            z = np.mod(x, y)  # expected output [2., -3.,  5., -2.,  3.,  3.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mod_float_mixed_sign_example')
        def _FMOD_MIXED_SIGN():
            node = onnx.helper.make_node(
                'Mod',
                inputs=['x', 'y'],
                outputs=['z'],
                fmod=1
            )
            
            x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
            y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])
            z = np.fmod(x, y)  # expected output [-0.1,  0.4,  5. ,  0.1, -0.4,  3.]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mod_fmod_mixed_sign_example')
        def _INT64_MIXED_SIGN():
            node = onnx.helper.make_node(
                'Mod',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
            y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
            z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mod_int64_mixed_sign_example')
        def _MUL_BROADCAST():
            node = onnx.helper.make_node(
                'Mod',
                inputs=['x', 'y'],
                outputs=['z'],
            )
            
            x = np.arange(0, 30).reshape([3, 2, 5])
            y = np.array([7])
            z = np.mod(x, y)
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_mod_bcast')
        _FLOAT_MIXED_SIGN()
        _FMOD_MIXED_SIGN()
        _INT64_MIXED_SIGN()
        _MUL_BROADCAST()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ThresholdedRelu
        self.run_ = nn._ThresholdedRelu_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.alpha, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _THRESHOLDEDRELU():
            alpha = 2.0
            node = onnx.helper.make_node(
                'ThresholdedRelu',
                inputs=['x'],
                outputs=['y'],
                alpha=alpha
            )
            
            x = np.array([-1.5, 0., 1.2, 2.0, 2.2]).astype(np.float32)
            y = np.clip(x, alpha, np.inf)  # expected output [0., 0., 0., 0., 2.2]
            y[y == alpha] = 0
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_thresholdedrelu_example')
            
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, alpha, np.inf)
            y[y == alpha] = 0
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_thresholdedrelu')
        def _DEFAULT():
            default_alpha = 1.0
            node = onnx.helper.make_node(
                'ThresholdedRelu',
                inputs=['x'],
                outputs=['y']
            )
            x = np.random.randn(3, 4, 5).astype(np.float32)
            y = np.clip(x, default_alpha, np.inf)
            y[y == default_alpha] = 0
            
            expect(node, inputs=[x], outputs=[y],
                   name='test_thresholdedrelu_default')
        _THRESHOLDEDRELU()
        _DEFAULT()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._MatMulInteger
        self.run_ = nn._MatMulInteger_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.A_i, self.B_i, self.a_zero_point_i, self.b_zero_point_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _MATMULINTEGER():
            node = onnx.helper.make_node('MatMulInteger',
                inputs=['A', 'B', 'a_zero_point', 'b_zero_point'],
                outputs=['Y'],)
            
            A = np.array([[11, 7, 3],
                [10, 6, 2],
                [9, 5, 1],
                [8, 4, 0], ], dtype=np.uint8)
            
            a_zero_point = np.array([12], dtype=np.uint8)
            
            B = np.array([[1, 4],
                [2, 5],
                [3, 6], ], dtype=np.uint8)
            
            b_zero_point = np.array([0], dtype=np.uint8)
            
            output = np.array([[-38, -83],
                [-44, -98],
                [-50, -113],
                [-56, -128], ], dtype=np.int32)
            
            expect(node, inputs=[A, B, a_zero_point, b_zero_point], outputs=[output],
                   name='test_matmulinteger')
        _MATMULINTEGER()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._QLinearMatMul
        self.run_ = nn._QLinearMatMul_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.a_i, self.a_scale_i, self.a_zero_point_i, self.b_i, self.b_scale_i, self.b_zero_point_i, self.y_scale_i, self.y_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _QLINEARMATMUL():
            node = onnx.helper.make_node('QLinearMatMul',
                inputs=['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
                outputs=['y'],)
            
            #2D
            a = np.array([[208, 236, 0, 238],
                [3, 214, 255, 29], ], dtype=np.uint8)
            
            a_scale = np.array([0.0066], dtype=np.float32)
            a_zero_point = np.array([113], dtype=np.uint8)
            
            b = np.array([[152, 51, 244],
                [60, 26, 255],
                [0, 127, 246],
                [127, 254, 247]], dtype=np.uint8)
            
            b_scale = np.array([0.00705], dtype=np.float32)
            b_zero_point = np.array([114], dtype=np.uint8)
            
            y_scale = np.array([0.0107], dtype=np.float32)
            y_zero_point = np.array([118], dtype=np.uint8)
            
            output = np.array([[168, 115, 255],
                [1, 66, 151], ], dtype=np.uint8)
            
            expect(node, inputs=[a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point], outputs=[output],
                   name='test_qlinearmatmul_2D')
            
            #3D
            a = np.array([[[208, 236, 0, 238],
                [3, 214, 255, 29]],
                [[208, 236, 0, 238],
                [3, 214, 255, 29]]], dtype=np.uint8)
            
            a_scale = np.array([0.0066], dtype=np.float32)
            a_zero_point = np.array([113], dtype=np.uint8)
            
            b = np.array([[[152, 51, 244],
                [60, 26, 255],
                [0, 127, 246],
                [127, 254, 247]],
                [[152, 51, 244],
                [60, 26, 255],
                [0, 127, 246],
                [127, 254, 247]]], dtype=np.uint8)
            
            b_scale = np.array([0.00705], dtype=np.float32)
            b_zero_point = np.array([114], dtype=np.uint8)
            
            y_scale = np.array([0.0107], dtype=np.float32)
            y_zero_point = np.array([118], dtype=np.uint8)
            
            output = np.array([[[168, 115, 255],
                [1, 66, 151]],
                [[168, 115, 255],
                [1, 66, 151]]], dtype=np.uint8)
            
            expect(node, inputs=[a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point], outputs=[output],
                   name='test_qlinearmatmul_3D')
        _QLINEARMATMUL()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ConvInteger
        self.run_ = nn._ConvInteger_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.x_i, self.w_i, self.x_zero_point_i, self.w_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _CONVINTEGER():
            
            x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
            x_zero_point = np.array([1]).astype(np.uint8)
            w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))
            
            y = np.array([12, 16, 24, 28]).astype(np.int32).reshape(1, 1, 2, 2)
            
            # ConvInteger without padding
            convinteger_node = onnx.helper.make_node('ConvInteger',
                inputs=['x', 'w', 'x_zero_point'],
                outputs=['y'])
            
            expect(convinteger_node, inputs=[x, w, x_zero_point], outputs=[y],
                   name='test_basic_convinteger')
            
            # ConvInteger with padding
            y_with_padding = np.array([1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9]).astype(np.int32).reshape((1, 1, 4, 4))
            
            convinteger_node_with_padding = onnx.helper.make_node('ConvInteger',
                inputs=['x', 'w', 'x_zero_point'],
                outputs=['y'],
                pads=[1, 1, 1, 1],)
            
            expect(convinteger_node_with_padding, inputs=[x, w, x_zero_point], outputs=[y_with_padding],
                   name='test_convinteger_with_padding')
        _CONVINTEGER()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._QLinearConv
        self.run_ = nn._QLinearConv_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides, self.x_i, self.x_scale_i, self.x_zero_point_i, self.w_i, self.w_scale_i, self.w_zero_point_i, self.y_scale_i, self.y_zero_point_i, self.B_i, self.y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _QLINEARCONV():
            node = onnx.helper.make_node('QLinearConv',
                inputs=['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'],
                outputs=['y'],)
            
            x = np.array([[255, 174, 162, 25, 203, 168, 58],
                [15, 59, 237, 95, 129, 0, 64],
                [56, 242, 153, 221, 168, 12, 166],
                [232, 178, 186, 195, 237, 162, 237],
                [188, 39, 124, 77, 80, 102, 43],
                [127, 230, 21, 83, 41, 40, 134],
                [255, 154, 92, 141, 42, 148, 247], ], dtype=np.uint8).reshape((1, 1, 7, 7))
            
            x_scale = np.array([0.00369204697], dtype=np.float32)
            x_zero_point = np.array([132], dtype=np.uint8)
            
            w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
            
            w_scale = np.array([0.00172794575], dtype=np.float32)
            w_zero_point = np.array([255], dtype=np.uint8)
            
            y_scale = np.array([0.00162681262], dtype=np.float32)
            y_zero_point = np.array([123], dtype=np.uint8)
            
            output = np.array([[0, 81, 93, 230, 52, 87, 197],
                [240, 196, 18, 160, 126, 255, 191],
                [199, 13, 102, 34, 87, 243, 89],
                [23, 77, 69, 60, 18, 93, 18],
                [67, 216, 131, 178, 175, 153, 212],
                [128, 25, 234, 172, 214, 215, 121],
                [0, 101, 163, 114, 213, 107, 8], ], dtype=np.uint8).reshape((1, 1, 7, 7))
            
            expect(node, inputs=[x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point], outputs=[output],
                   name='test_qlinearconv')
        _QLINEARCONV()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._QuantizeLinear
        self.run_ = nn._QuantizeLinear_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.x_i, self.y_scale_i, self.y_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _QUANTIZELINEAR():
            node = onnx.helper.make_node('QuantizeLinear',
                inputs=['x', 'y_scale', 'y_zero_point'],
                outputs=['y'],)
            
            x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
            y_scale = np.array([2], dtype=np.float32)
            y_zero_point = np.array([128], dtype=np.uint8)
            y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)
            
            expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
                   name='test_quantizelinear')
        _QUANTIZELINEAR()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._DequantizeLinear
        self.run_ = nn._DequantizeLinear_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.x_i, self.x_scale_i, self.x_zero_point_i, self.y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _DEQUANTIZELINEAR():
            node = onnx.helper.make_node('DequantizeLinear',
                inputs=['x', 'x_scale', 'x_zero_point'],
                outputs=['y'],)
            
            # scalar zero point and scale
            x = np.array([0, 3, 128, 255]).astype(np.uint8)
            x_scale = np.array([2], dtype=np.float32)
            x_zero_point = np.array([128], dtype=np.uint8)
            y = np.array([-256, -250, 0, 254], dtype=np.float32)
            
            expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
                   name='test_dequantizelinear')
        _DEQUANTIZELINEAR()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._IsInf
        self.run_ = nn._IsInf_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.detect_negative, self.detect_positive, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _INFINITY():
            node = onnx.helper.make_node('IsInf',
                                         inputs=['x'],
                                         outputs=['y'],
                                         )
            
            x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                         dtype=np.float32)
            y = np.isinf(x)
            expect(node, inputs=[x], outputs=[y], name='test_isinf')
        def _POSITIVE_INFINITY_ONLY():
            node = onnx.helper.make_node('IsInf',
                                         inputs=['x'],
                                         outputs=['y'],
                                         detect_negative=0
                                         )
            
            x = np.array([-1.7, np.nan, np.inf, 3.6, np.NINF, np.inf],
                         dtype=np.float32)
            y = np.isposinf(x)
            expect(node, inputs=[x], outputs=[y], name='test_isinf_positive')
        def _NEGATIVE_INFINITY_ONLY():
            node = onnx.helper.make_node('IsInf',
                                         inputs=['x'],
                                         outputs=['y'],
                                         detect_positive=0
                                         )
            
            x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
                         dtype=np.float32)
            y = np.isneginf(x)
            expect(node, inputs=[x], outputs=[y], name='test_isinf_negative')
        _INFINITY()
        _POSITIVE_INFINITY_ONLY()
        _NEGATIVE_INFINITY_ONLY()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._RoiAlign
        self.run_ = nn._RoiAlign_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.mode, self.output_height, self.output_width, self.sampling_ratio, self.spatial_scale, self.X_i, self.rois_i, self.batch_indices_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):

        def _ROIALIGN():
            node = onnx.helper.make_node(
                "RoiAlign",
                inputs=["X", "rois", "batch_indices"],
                outputs=["Y"],
                spatial_scale=1.0,
                output_height=5,
                output_width=5,
                sampling_ratio=2,
            )
            
            X = np.array(
                [
                    [
                        [
                            [
                                0.2764,
                                0.7150,
                                0.1958,
                                0.3416,
                                0.4638,
                                0.0259,
                                0.2963,
                                0.6518,
                                0.4856,
                                0.7250,
                            ],
                            [
                                0.9637,
                                0.0895,
                                0.2919,
                                0.6753,
                                0.0234,
                                0.6132,
                                0.8085,
                                0.5324,
                                0.8992,
                                0.4467,
                            ],
                            [
                                0.3265,
                                0.8479,
                                0.9698,
                                0.2471,
                                0.9336,
                                0.1878,
                                0.4766,
                                0.4308,
                                0.3400,
                                0.2162,
                            ],
                            [
                                0.0206,
                                0.1720,
                                0.2155,
                                0.4394,
                                0.0653,
                                0.3406,
                                0.7724,
                                0.3921,
                                0.2541,
                                0.5799,
                            ],
                            [
                                0.4062,
                                0.2194,
                                0.4473,
                                0.4687,
                                0.7109,
                                0.9327,
                                0.9815,
                                0.6320,
                                0.1728,
                                0.6119,
                            ],
                            [
                                0.3097,
                                0.1283,
                                0.4984,
                                0.5068,
                                0.4279,
                                0.0173,
                                0.4388,
                                0.0430,
                                0.4671,
                                0.7119,
                            ],
                            [
                                0.1011,
                                0.8477,
                                0.4726,
                                0.1777,
                                0.9923,
                                0.4042,
                                0.1869,
                                0.7795,
                                0.9946,
                                0.9689,
                            ],
                            [
                                0.1366,
                                0.3671,
                                0.7011,
                                0.6234,
                                0.9867,
                                0.5585,
                                0.6985,
                                0.5609,
                                0.8788,
                                0.9928,
                            ],
                            [
                                0.5697,
                                0.8511,
                                0.6711,
                                0.9406,
                                0.8751,
                                0.7496,
                                0.1650,
                                0.1049,
                                0.1559,
                                0.2514,
                            ],
                            [
                                0.7012,
                                0.4056,
                                0.7879,
                                0.3461,
                                0.0415,
                                0.2998,
                                0.5094,
                                0.3727,
                                0.5482,
                                0.0502,
                            ],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
            batch_indices = np.array([0, 0, 0], dtype=np.int64)
            rois = np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)
            # (num_rois, C, output_height, output_width)
            Y = np.array(
                [
                    [
                        [
                            [0.4664, 0.4466, 0.3405, 0.5688, 0.6068],
                            [0.3714, 0.4296, 0.3835, 0.5562, 0.3510],
                            [0.2768, 0.4883, 0.5222, 0.5528, 0.4171],
                            [0.4713, 0.4844, 0.6904, 0.4920, 0.8774],
                            [0.6239, 0.7125, 0.6289, 0.3355, 0.3495],
                        ]
                    ],
                    [
                        [
                            [0.3022, 0.4305, 0.4696, 0.3978, 0.5423],
                            [0.3656, 0.7050, 0.5165, 0.3172, 0.7015],
                            [0.2912, 0.5059, 0.6476, 0.6235, 0.8299],
                            [0.5916, 0.7389, 0.7048, 0.8372, 0.8893],
                            [0.6227, 0.6153, 0.7097, 0.6154, 0.4585],
                        ]
                    ],
                    [
                        [
                            [0.2384, 0.3379, 0.3717, 0.6100, 0.7601],
                            [0.3767, 0.3785, 0.7147, 0.9243, 0.9727],
                            [0.5749, 0.5826, 0.5709, 0.7619, 0.8770],
                            [0.5355, 0.2566, 0.2141, 0.2796, 0.3600],
                            [0.4365, 0.3504, 0.2887, 0.3661, 0.2349],
                        ]
                    ],
                ],
                dtype=np.float32,
            )
            
            expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign")
        _ROIALIGN()

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ArrayFeatureExtractor
        self.run_ = nn._ArrayFeatureExtractor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.X_i, self.Y_i, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Binarizer
        self.run_ = nn._Binarizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.threshold, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._CategoryMapper
        self.run_ = nn._CategoryMapper_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.cats_int64s, self.cats_strings, self.default_int64, self.default_string, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._DictVectorizer
        self.run_ = nn._DictVectorizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.int64_vocabulary, self.string_vocabulary, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['DictVectorizer'] = DictVectorizer





class FeatureVectorizer:
    name = None
    Y_o = str()

    #parameters
    inputdimensions = list()

    input_params = []
    output_params = ["Y_o"]
    attribute_params = ["inputdimensions"]
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._FeatureVectorizer
        self.run_ = nn._FeatureVectorizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.inputdimensions, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LabelEncoder
        self.run_ = nn._LabelEncoder_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.default_float, self.default_int64, self.default_string, self.keys_floats, self.keys_int64s, self.keys_strings, self.values_floats, self.values_int64s, self.values_strings, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LinearClassifier
        self.run_ = nn._LinearClassifier_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.coefficients, self.classlabels_ints, self.classlabels_strings, self.intercepts, self.multi_class, self.post_transform, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._LinearRegressor
        self.run_ = nn._LinearRegressor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.coefficients, self.intercepts, self.post_transform, self.targets, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Normalizer
        self.run_ = nn._Normalizer_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.norm, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._SVMRegressor
        self.run_ = nn._SVMRegressor_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.coefficients, self.kernel_params, self.kernel_type, self.n_supports, self.one_class, self.post_transform, self.rho, self.support_vectors, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._Scaler
        self.run_ = nn._Scaler_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.offset, self.scale, self.X_i, self.Y_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._TreeEnsembleClassifier
        self.run_ = nn._TreeEnsembleClassifier_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.base_values, self.class_ids, self.class_nodeids, self.class_treeids, self.class_weights, self.classlabels_int64s, self.classlabels_strings, self.nodes_falsenodeids, self.nodes_featureids, self.nodes_hitrates, self.nodes_missing_value_tracks_true, self.nodes_modes, self.nodes_nodeids, self.nodes_treeids, self.nodes_truenodeids, self.nodes_values, self.post_transform, self.X_i, self.Y_o, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

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
  
    def __init__(self, name, **kwargs):
        self.name = name
        self.Module = nn._ZipMap
        self.run_ = nn._ZipMap_run
        self.__dict__.update(kwargs)
       
    def output_shape(self):
        return tensors[self.__dict__[self.input_params[0]]].shape 

    def __call__(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.input_params[i]] = x           
        return self

    def output(self, *args):
        for i, x in enumerate(args):
            self.__dict__[self.output_params[i]] = x            
            if(x not in tensors.keys()):     
                tensors[x] =  np.zeros(self.output_shape())
        return self

    def build(self):
        self.Module(self.name, self.classlabels_int64s, self.classlabels_strings, self.X_i, self.Z_o)

    def run(self):
        self.run_(self.name)

    def test(self):
        pass

layer_map['ZipMap'] = ZipMap

