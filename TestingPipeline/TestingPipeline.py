import numpy as np
import _backend as backend


if(__name__=="__main__"):
    x = np.ones(128)
    backend.test()
    backend.create_instance()
    backend.input(x)
