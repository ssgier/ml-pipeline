from unittest import TestCase
import numpy as np
from ml_pipeline.layer_nm_model import (
    LayerNMModel,
    ConvolutionLayer2DConfig,
    FullyConnectedLayerConfig,
)
from ml_pipeline.convolution_2d_ff_nm_model import (
    Convolution2DFFNMModel,
    Config as FFConfig,
)
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)


class TestConvolution2DFFNMModel(TestCase):
    def simple_scenario(self):
        layer_1_config = ConvolutionLayer2DConfig(
            in_size=5,
            conv_kernel_width=3,
            conv_stride=2,
            ltp_step_up=0.3,
            ltp_step_down=0.34,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0,
            inhibition_reach=1,
            num_out_spikes=3,
        )

        layer_2_config = FullyConnectedLayerConfig(
            N_in=9,
            N_out=5,
            ltp_step_up=0.3,
            ltp_step_down=0.36,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0,
            num_out_spikes=1,
        )

        ff_config = FFConfig(N_repeat=1)

        ff_model = Convolution2DFFNMModel(
            ff_config, [LayerNMModel(layer_1_config), LayerNMModel(layer_2_config)]
        )

        ff_model._layers[0]._weights[1, 2] = 0.6
        ff_model._layers[0]._weights[2, 0] = 0.7

        # TODO: define composite result, containing internal info about layer activations
