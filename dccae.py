import numpy as np
from torch import optim
from cca_zoo import data
import scipy.io as sio
from cca_zoo.deepmodels import DCCA, DCCAE, DVCCA, DCCA_NOI, DTCCA, SplitAE, DeepWrapper
from cca_zoo.deepmodels import objectives, architectures
filename = "testdata"
data=sio.loadmat(filename)
latent_dims = 2048
device = 'cuda'
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=2048,layer_sizes=[4096,4096,4096])
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=2048,layer_sizes=[4096,4096,4096])
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=2048,layer_sizes=[4096,4096,4096])
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=2048,layer_sizes=[4096,4096,4096])
# DCCAE
dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                    decoders=[decoder_1, decoder_2], objective=objectives.CCA,lam=1e-3)
# hidden_layer_sizes are shown explicitly but these are also the defaults
dccae_model = DeepWrapper(dccae_model, device=device)
dccae_model.fit((data['XV1'], data['XV2']),epochs=50)