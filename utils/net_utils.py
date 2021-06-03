from core.dgp.dense_layer import SVGP_Layer, SVGP_KFGD_Layer
from core.dgp.blk_layer import BlockSVGP_KFGD_Layer
from core.dgp.conv_layer import Conv_SVGP_Layer, Conv_KFGD_Layer
from core.dgp.blk_conv_layer import BlockConvKFGD_Layer
from core.dgp.utils.feature import PatchInducingFeatures, PatchLocInducingFeatures
from core.dgp.utils.feature2 import MirrorTwoFeaturesV3, MirrorTwoLocFeaturesV3
from core.dgp.blk_layer import SOLVEGP_KFGD_Layer
from core.dgp.blk_conv_layer import SOLVEGP_CONV_KFGD_Layer

from core.dgp.multigpu.dgp import MultiGPU_DGP
from core.dgp.dgp import DGP
from core.dgp.multigpu.feature import SingleMirrorFourFeatures
from core.dgp.multigpu.feature import SingleMirrorFourLocFeatures
from core.dgp.multigpu.blk_layer import MultiGPU_KFGD_Layer
from core.dgp.multigpu.blk_conv_layer import MultiGPU_ConvKFGD_Layer


def networks(name):

    if name == 'L1-1':
        return dict(
            kerns=['tick'],
            feature_maps=[],
            filter_sizes=[5],
            strides=[1],
            pools=[],
            pool_sizes=[],
            nms=[1000],
            paddings=["VALID"],
            inducing_features=[PatchLocInducingFeatures]
        )
    if name == 'L1-2':
        return dict(
            kerns=['tick'],
            feature_maps=[],
            filter_sizes=[5],
            strides=[1],
            pools=[],
            pool_sizes=[],
            nms=[2000],
            paddings=["VALID"],
            inducing_features=[PatchLocInducingFeatures]
        )
    if name == 'L1-1-2-v3':
        return dict(
            kerns=['tick'],
            feature_maps=[],
            filter_sizes=[5],
            strides=[1],
            pools=[],
            pool_sizes=[],
            nms=[1000],
            paddings=["VALID"],
            inducing_features=[MirrorTwoLocFeaturesV3]
        )
    if name == 'L1-1-mg4':
        return dict(
            kerns=['tick'],
            feature_maps=[],
            filter_sizes=[5],
            strides=[1],
            pools=[],
            pool_sizes=[],
            nms=[1000],
            paddings=["VALID"],
            inducing_features=[SingleMirrorFourLocFeatures]
        )


    if name == 'L2-3':
        return dict(
            kerns=['rbf', 'tick'],
            feature_maps=[10],
            filter_sizes=[5, 4],
            strides=[1, 2],
            pools=[None],
            pool_sizes=[1],
            nms=[384, 1000],
            paddings=["VALID", "VALID"],
            inducing_features=[PatchInducingFeatures, PatchLocInducingFeatures]
        )
    if name == 'L2-4':
        return dict(
            kerns=['rbf', 'tick'],
            feature_maps=[10],
            filter_sizes=[5, 4],
            strides=[1, 2],
            pools=[None],
            pool_sizes=[1],
            nms=[768, 2000],
            paddings=["VALID", "VALID"],
            inducing_features=[PatchInducingFeatures, PatchLocInducingFeatures]
        )
    if name == 'L2-3-2-v3':
        return dict(
            kerns=['rbf', 'tick'],
            feature_maps=[10],
            filter_sizes=[5, 4],
            strides=[1, 2],
            pools=[None],
            pool_sizes=[1],
            nms=[384, 1000],
            paddings=["VALID", "VALID"],
            inducing_features=[MirrorTwoFeaturesV3, MirrorTwoLocFeaturesV3]
        )
    if name == 'L2-3-mg4':
        return dict(
            kerns=['rbf', 'tick'],
            feature_maps=[10],
            filter_sizes=[5, 4],
            strides=[1, 2],
            pools=[None],
            pool_sizes=[1],
            nms=[384, 1000],
            paddings=["VALID", "VALID"],
            inducing_features=[SingleMirrorFourFeatures, SingleMirrorFourLocFeatures]
        )


    if name == 'L3-4':
        return dict(
            kerns=['rbf', 'rbf', 'tick'],
            feature_maps=[10, 10],
            filter_sizes=[5, 4, 5],
            strides=[1, 2, 1],
            pools=[None, None],
            pool_sizes=[1, 1],
            nms=[384, 384, 1000],
            paddings=["VALID", "VALID", "VALID"],
            inducing_features=[PatchInducingFeatures, PatchInducingFeatures, PatchLocInducingFeatures],
        )
    if name == 'L3-21':
        return dict(
            kerns=['rbf', 'rbf', 'tick'],
            feature_maps=[10, 10],
            filter_sizes=[5, 4, 5],
            strides=[1, 2, 1],
            pools=[None, None],
            pool_sizes=[1, 1],
            nms=[768, 768, 2000],
            paddings=["VALID", "VALID", "VALID"],
            inducing_features=[PatchInducingFeatures, PatchInducingFeatures, PatchLocInducingFeatures],
         )
    if name == 'L3-4-2-v3':
        return dict(
            kerns=['rbf', 'rbf', 'tick'],
            feature_maps=[10, 10],
            filter_sizes=[5, 4, 5],
            strides=[1, 2, 1],
            pools=[None, None],
            pool_sizes=[1, 1],
            nms=[384, 384, 1000],
            paddings=["VALID", "VALID", "VALID"],
            inducing_features=[MirrorTwoFeaturesV3, MirrorTwoFeaturesV3, MirrorTwoLocFeaturesV3],
        )
    if name == 'L3-4-mg4':
        return dict(
            kerns=['rbf', 'rbf', 'tick'],
            feature_maps=[10, 10],
            filter_sizes=[5, 4, 5],
            strides=[1, 2, 1],
            pools=[None, None],
            pool_sizes=[1, 1],
            nms=[384, 384, 1000],
            paddings=["VALID", "VALID", "VALID"],
            inducing_features=[SingleMirrorFourFeatures, SingleMirrorFourFeatures, SingleMirrorFourLocFeatures],
        )


    if name == 'L4-7':
        return dict(
            kerns=['rbf', 'rbf', 'rbf', 'tick'],
            feature_maps=[16, 16, 16],
            filter_sizes=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
            pools=[None, 'mean', 'mean'],
            pool_sizes=[1, 2, 1],
            nms=[384, 384, 384, 1000],
            paddings=["SAME", "SAME", "SAME", "SAME"],
            inducing_features=[PatchInducingFeatures, PatchInducingFeatures,
                               PatchInducingFeatures, PatchLocInducingFeatures],
        )
    if name == 'L4-22':
        return dict(
            kerns=['rbf', 'rbf', 'rbf', 'tick'],
            feature_maps=[16, 16, 16],
            filter_sizes=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
            pools=[None, 'mean', 'mean'],
            pool_sizes=[1, 2, 1],
            nms=[700, 700, 700, 1600],
            paddings=["SAME", "SAME", "SAME", "SAME"],
            inducing_features=[PatchInducingFeatures, PatchInducingFeatures,
                               PatchInducingFeatures, PatchLocInducingFeatures],
        )
    if name == 'L4-7-2-v3':
        return dict(
            kerns=['rbf', 'rbf', 'rbf', 'tick'],
            feature_maps=[16, 16, 16],
            filter_sizes=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
            pools=[None, 'mean', 'mean'],
            pool_sizes=[1, 2, 1],
            nms=[384, 384, 384, 1000],
            paddings=["SAME", "SAME", "SAME", "SAME"],
            inducing_features=[MirrorTwoFeaturesV3, MirrorTwoFeaturesV3,
                               MirrorTwoFeaturesV3, MirrorTwoLocFeaturesV3],
        )
    if name == 'L4-7-mg4':
        return dict(
            kerns=['rbf', 'rbf', 'rbf', 'tick'],
            feature_maps=[16, 16, 16],
            filter_sizes=[5, 4, 5, 4],
            strides=[1, 1, 1, 1],
            pools=[None, 'mean', 'mean'],
            pool_sizes=[1, 2, 1],
            nms=[384, 384, 384, 1000],
            paddings=["SAME", "SAME", "SAME", "SAME"],
            inducing_features=[SingleMirrorFourFeatures, SingleMirrorFourFeatures,
                               SingleMirrorFourFeatures, SingleMirrorFourLocFeatures],
        )


DENSE_LAYER_DICT =  dict(
    dsvi=SVGP_Layer,
    kfgd=SVGP_KFGD_Layer,
    sovgd=SOLVEGP_KFGD_Layer,
    blkkfgd=BlockSVGP_KFGD_Layer,
    mgkfgd=MultiGPU_KFGD_Layer,
)

CONV_LAYER_DICT =  dict(
    dsvi=Conv_SVGP_Layer,
    kfgd=Conv_KFGD_Layer,
    sovgd=SOLVEGP_CONV_KFGD_Layer,
    blkkfgd=BlockConvKFGD_Layer,
    mgkfgd=MultiGPU_ConvKFGD_Layer,
)

MODEL_DICT = dict(
    dsvi=DGP,
    kfgd=DGP,
    sovgd=DGP,
    blkkfgd=DGP,
    mgkfgd=MultiGPU_DGP
)
