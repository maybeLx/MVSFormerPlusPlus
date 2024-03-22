import numpy as np

resolutions = np.array([[512, 640], [512, 704],[512, 768],
                        [576, 704],[576, 768],[576, 832],
                        [640, 832], [640, 896], [640, 960],
                        [704, 896], [704, 960], [704, 1024],
                        [768, 960], [768, 1024], [768, 1088],
                        [832, 1024], [832, 1088], [832, 1152],
                        [896, 1152], [896, 1216], [896, 1280],
                        [960, 1216 ], [960, 1280], [960, 1344],
                        [1024, 1280]])
# whole_length = 0.
pure_transformer_length = 0.
for resolution in resolutions:
    length = resolution[0] /8 * resolution[1] / 8
    pure_transformer_length+=length

print(pure_transformer_length/resolutions.shape[0])
