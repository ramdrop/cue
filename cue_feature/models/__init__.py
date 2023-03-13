from logging import raiseExceptions
import models.resunet as resunets
import models.ddpnet as ddpnets
import models.mgnet as mgnet

import matin


log = matin.ln(__name__).get_logger()

MODEL_LIST = []
def add_models(module):
    MODEL_LIST.extend([getattr(module, a) for a in dir(module) if 'Net' in a or 'MLP' in a])

add_models(resunets)
add_models(ddpnets)
add_models(mgnet)

def load_model(name):
    '''
    Creates and returns an instance of the model given its class name.
    '''
    MODEL_LUT = {model.__name__: model for model in MODEL_LIST}
    if name not in MODEL_LUT:
        raiseExceptions(f'Undefined model name: {name}')
    Model = MODEL_LUT[name]

    return Model
