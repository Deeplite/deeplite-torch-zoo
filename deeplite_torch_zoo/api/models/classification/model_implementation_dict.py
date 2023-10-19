# pylint: disable=too-many-lines

MODEL_IMPLEMENTATIONS = {
    'torchvision': [
        'alexnet',
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'resnext50_32x4d',
        'resnext101_32x8d',
        'wide_resnet50_2',
        'wide_resnet101_2',
        'vgg11',
        'vgg11_bn',
        'vgg13',
        'vgg13_bn',
        'vgg16',
        'vgg16_bn',
        'vgg19_bn',
        'vgg19',
        'squeezenet1_0',
        'squeezenet1_1',
        'inception_v3',
        'densenet121',
        'densenet169',
        'densenet201',
        'densenet161',
        'googlenet',
        'mobilenet_v2',
        'mobilenet_v3_large',
        'mobilenet_v3_small',
        'mnasnet0_5',
        'mnasnet0_75',
        'mnasnet1_0',
        'mnasnet1_3',
        'shufflenet_v2_x0_5',
        'shufflenet_v2_x1_0',
        'shufflenet_v2_x1_5',
        'shufflenet_v2_x2_0',
    ],
    'pytorchcv': [
        'alexnet',
        'alexnetb',
        'zfnet',
        'zfnetb',
        'vgg11',
        'vgg13',
        'vgg16',
        'vgg19',
        'bn_vgg11',
        'bn_vgg13',
        'bn_vgg16',
        'bn_vgg19',
        'bn_vgg11b',
        'bn_vgg13b',
        'bn_vgg16b',
        'bn_vgg19b',
        'bninception',
        'resnet10',
        'resnet12',
        'resnet14',
        'resnetbc14b',
        'resnet16',
        'resnet18_wd4',
        'resnet18_wd2',
        'resnet18_w3d4',
        'resnet18',
        'resnet26',
        'resnetbc26b',
        'resnet34',
        'resnetbc38b',
        'resnet50',
        'resnet50b',
        'resnet101',
        'resnet101b',
        'resnet152',
        'resnet152b',
        'resnet200',
        'resnet200b',
        'preresnet10',
        'preresnet12',
        'preresnet14',
        'preresnetbc14b',
        'preresnet16',
        'preresnet18_wd4',
        'preresnet18_wd2',
        'preresnet18_w3d4',
        'preresnet18',
        'preresnet26',
        'preresnetbc26b',
        'preresnet34',
        'preresnetbc38b',
        'preresnet50',
        'preresnet50b',
        'preresnet101',
        'preresnet101b',
        'preresnet152',
        'preresnet152b',
        'preresnet200',
        'preresnet200b',
        'preresnet269b',
        'resnext14_16x4d',
        'resnext14_32x2d',
        'resnext14_32x4d',
        'resnext26_16x4d',
        'resnext26_32x2d',
        'resnext26_32x4d',
        'resnext38_32x4d',
        'resnext50_32x4d',
        'resnext101_32x4d',
        'resnext101_64x4d',
        'seresnet10',
        'seresnet12',
        'seresnet14',
        'seresnet16',
        'seresnet18',
        'seresnet26',
        'seresnetbc26b',
        'seresnet34',
        'seresnetbc38b',
        'seresnet50',
        'seresnet50b',
        'seresnet101',
        'seresnet101b',
        'seresnet152',
        'seresnet152b',
        'seresnet200',
        'seresnet200b',
        'sepreresnet10',
        'sepreresnet12',
        'sepreresnet14',
        'sepreresnet16',
        'sepreresnet18',
        'sepreresnet26',
        'sepreresnetbc26b',
        'sepreresnet34',
        'sepreresnetbc38b',
        'sepreresnet50',
        'sepreresnet50b',
        'sepreresnet101',
        'sepreresnet101b',
        'sepreresnet152',
        'sepreresnet152b',
        'sepreresnet200',
        'sepreresnet200b',
        'seresnext50_32x4d',
        'seresnext101_32x4d',
        'seresnext101_64x4d',
        'senet16',
        'senet28',
        'senet40',
        'senet52',
        'senet103',
        'senet154',
        'resnestabc14',
        'resnesta18',
        'resnestabc26',
        'resnesta50',
        'resnesta101',
        'resnesta152',
        'resnesta200',
        'resnesta269',
        'ibn_resnet50',
        'ibn_resnet101',
        'ibn_resnet152',
        'ibnb_resnet50',
        'ibnb_resnet101',
        'ibnb_resnet152',
        'ibn_resnext50_32x4d',
        'ibn_resnext101_32x4d',
        'ibn_resnext101_64x4d',
        'ibn_densenet121',
        'ibn_densenet161',
        'ibn_densenet169',
        'ibn_densenet201',
        'airnet50_1x64d_r2',
        'airnet50_1x64d_r16',
        'airnet101_1x64d_r2',
        'airnext50_32x4d_r2',
        'airnext101_32x4d_r2',
        'airnext101_32x4d_r16',
        'bam_resnet18',
        'bam_resnet34',
        'bam_resnet50',
        'bam_resnet101',
        'bam_resnet152',
        'cbam_resnet18',
        'cbam_resnet34',
        'cbam_resnet50',
        'cbam_resnet101',
        'cbam_resnet152',
        'resattnet56',
        'resattnet92',
        'resattnet128',
        'resattnet164',
        'resattnet200',
        'resattnet236',
        'resattnet452',
        'sknet50',
        'sknet101',
        'sknet152',
        'scnet50',
        'scnet101',
        'scneta50',
        'scneta101',
        'regnetx002',
        'regnetx004',
        'regnetx006',
        'regnetx008',
        'regnetx016',
        'regnetx032',
        'regnetx040',
        'regnetx064',
        'regnetx080',
        'regnetx120',
        'regnetx160',
        'regnetx320',
        'regnety002',
        'regnety004',
        'regnety006',
        'regnety008',
        'regnety016',
        'regnety032',
        'regnety040',
        'regnety064',
        'regnety080',
        'regnety120',
        'regnety160',
        'regnety320',
        'diaresnet10',
        'diaresnet12',
        'diaresnet14',
        'diaresnetbc14b',
        'diaresnet16',
        'diaresnet18',
        'diaresnet26',
        'diaresnetbc26b',
        'diaresnet34',
        'diaresnetbc38b',
        'diaresnet50',
        'diaresnet50b',
        'diaresnet101',
        'diaresnet101b',
        'diaresnet152',
        'diaresnet152b',
        'diaresnet200',
        'diaresnet200b',
        'diapreresnet10',
        'diapreresnet12',
        'diapreresnet14',
        'diapreresnetbc14b',
        'diapreresnet16',
        'diapreresnet18',
        'diapreresnet26',
        'diapreresnetbc26b',
        'diapreresnet34',
        'diapreresnetbc38b',
        'diapreresnet50',
        'diapreresnet50b',
        'diapreresnet101',
        'diapreresnet101b',
        'diapreresnet152',
        'diapreresnet152b',
        'diapreresnet200',
        'diapreresnet200b',
        'diapreresnet269b',
        'pyramidnet101_a360',
        'diracnet18v2',
        'diracnet34v2',
        'sharesnet18',
        'sharesnet34',
        'sharesnet50',
        'sharesnet50b',
        'sharesnet101',
        'sharesnet101b',
        'sharesnet152',
        'sharesnet152b',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201',
        'condensenet74_c4_g4',
        'condensenet74_c8_g8',
        'sparsenet121',
        'sparsenet161',
        'sparsenet169',
        'sparsenet201',
        'sparsenet264',
        'peleenet',
        'wrn50_2',
        'drnc26',
        'drnc42',
        'drnc58',
        'drnd22',
        'drnd38',
        'drnd54',
        'drnd105',
        'dpn68',
        'dpn68b',
        'dpn98',
        'dpn107',
        'dpn131',
        'darknet_ref',
        'darknet_tiny',
        'darknet19',
        'darknet53',
        'channelnet',
        'revnet38',
        'revnet110',
        'revnet164',
        'irevnet301',
        'bagnet9',
        'bagnet17',
        'bagnet33',
        'dla34',
        'dla46c',
        'dla46xc',
        'dla60',
        'dla60x',
        'dla60xc',
        'dla102',
        'dla102x',
        'dla102x2',
        'dla169',
        'msdnet22',
        'fishnet99',
        'fishnet150',
        'espnetv2_wd2',
        'espnetv2_w1',
        'espnetv2_w5d4',
        'espnetv2_w3d2',
        'espnetv2_w2',
        'dicenet_wd5',
        'dicenet_wd2',
        'dicenet_w3d4',
        'dicenet_w1',
        'dicenet_w5d4',
        'dicenet_w3d2',
        'dicenet_w7d8',
        'dicenet_w2',
        'hrnet_w18_small_v1',
        'hrnet_w18_small_v2',
        'hrnetv2_w18',
        'hrnetv2_w30',
        'hrnetv2_w32',
        'hrnetv2_w40',
        'hrnetv2_w44',
        'hrnetv2_w48',
        'hrnetv2_w64',
        'vovnet27s',
        'vovnet39',
        'vovnet57',
        'selecsls42',
        'selecsls42b',
        'selecsls60',
        'selecsls60b',
        'selecsls84',
        'hardnet39ds',
        'hardnet68ds',
        'hardnet68',
        'hardnet85',
        'xdensenet121_2',
        'xdensenet161_2',
        'xdensenet169_2',
        'xdensenet201_2',
        'squeezenet_v1_0',
        'squeezenet_v1_1',
        'squeezeresnet_v1_0',
        'squeezeresnet_v1_1',
        'sqnxt23_w1',
        'sqnxt23_w3d2',
        'sqnxt23_w2',
        'sqnxt23v5_w1',
        'sqnxt23v5_w3d2',
        'sqnxt23v5_w2',
        'shufflenet_g1_w1',
        'shufflenet_g2_w1',
        'shufflenet_g3_w1',
        'shufflenet_g4_w1',
        'shufflenet_g8_w1',
        'shufflenet_g1_w3d4',
        'shufflenet_g3_w3d4',
        'shufflenet_g1_wd2',
        'shufflenet_g3_wd2',
        'shufflenet_g1_wd4',
        'shufflenet_g3_wd4',
        'shufflenetv2_wd2',
        'shufflenetv2_w1',
        'shufflenetv2_w3d2',
        'shufflenetv2_w2',
        'shufflenetv2b_wd2',
        'shufflenetv2b_w1',
        'shufflenetv2b_w3d2',
        'shufflenetv2b_w2',
        'menet108_8x1_g3',
        'menet128_8x1_g4',
        'menet160_8x1_g8',
        'menet228_12x1_g3',
        'menet256_12x1_g4',
        'menet348_12x1_g3',
        'menet352_12x1_g8',
        'menet456_24x1_g3',
        'mobilenet_w1',
        'mobilenet_w3d4',
        'mobilenet_wd2',
        'mobilenet_wd4',
        'mobilenetb_w1',
        'mobilenetb_w3d4',
        'mobilenetb_wd2',
        'mobilenetb_wd4',
        'fdmobilenet_w1',
        'fdmobilenet_w3d4',
        'fdmobilenet_wd2',
        'fdmobilenet_wd4',
        'mobilenetv2_w1',
        'mobilenetv2_w3d4',
        'mobilenetv2_wd2',
        'mobilenetv2_wd4',
        'mobilenetv2b_w1',
        'mobilenetv2b_w3d4',
        'mobilenetv2b_wd2',
        'mobilenetv2b_wd4',
        'mobilenetv3_small_w7d20',
        'mobilenetv3_small_wd2',
        'mobilenetv3_small_w3d4',
        'mobilenetv3_small_w1',
        'mobilenetv3_small_w5d4',
        'mobilenetv3_large_w7d20',
        'mobilenetv3_large_wd2',
        'mobilenetv3_large_w3d4',
        'mobilenetv3_large_w1',
        'mobilenetv3_large_w5d4',
        'igcv3_w1',
        'igcv3_w3d4',
        'igcv3_wd2',
        'igcv3_wd4',
        'ghostnet',
        'mnasnet_b1',
        'mnasnet_a1',
        'mnasnet_small',
        'darts',
        'proxylessnas_cpu',
        'proxylessnas_gpu',
        'proxylessnas_mobile',
        'proxylessnas_mobile14',
        'fbnet_cb',
        'nasnet_4a1056',
        'spnasnet',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4',
        'efficientnet_b5',
        'efficientnet_b6',
        'efficientnet_b7',
        'efficientnet_b8',
        'efficientnet_b0b',
        'efficientnet_b1b',
        'efficientnet_b2b',
        'efficientnet_b3b',
        'efficientnet_b4b',
        'efficientnet_b5b',
        'efficientnet_b6b',
        'efficientnet_b7b',
        'efficientnet_b0c',
        'efficientnet_b1c',
        'efficientnet_b2c',
        'efficientnet_b3c',
        'efficientnet_b4c',
        'efficientnet_b5c',
        'efficientnet_b6c',
        'efficientnet_b7c',
        'efficientnet_b8c',
        'efficientnet_edge_small_b',
        'efficientnet_edge_medium_b',
        'efficientnet_edge_large_b',
        'mixnet_s',
        'mixnet_m',
        'mixnet_l',
        'isqrtcovresnet18',
        'isqrtcovresnet34',
        'isqrtcovresnet50',
        'isqrtcovresnet50b',
        'isqrtcovresnet101',
        'isqrtcovresnet101b',
        'resneta10',
        'resnetabc14b',
        'resneta18',
        'resneta50b',
        'resneta101b',
        'resneta152b',
        'resnetd50b',
        'resnetd101b',
        'resnetd152b',
        'fastseresnet101b',
        'octresnet10_ad2',
        'octresnet50b_ad2',
    ],
}


# models that contain the inplace-ABN module
INPLACE_ABN_MODELS = [
    'densenet264d_iabn',
    'ese_vovnet99b_iabn',
    'tresnet_l_448',
    'tresnet_l',
    'tresnet_m_448',
    'tresnet_m',
    'tresnet_m_miil_in21k',
    'tresnet_v2_l',
    'tresnet_xl_448',
    'tresnet_xl',
]


# models that don't support 224x224 input image size
FIXED_SIZE_INPUT_MODELS = [
    'bat_resnext26ts',
    'beit_base_patch16_384',
    'beit_large_patch16_384',
    'beit_large_patch16_512',
    'botnet26t_256',
    'botnet50ts_256',
    'cait_m36_384',
    'cait_m48_448',
    'cait_s24_384',
    'cait_s36_384',
    'cait_xs24_384',
    'cait_xxs24_384',
    'cait_xxs36_384',
    'deit3_base_patch16_384',
    'deit3_base_patch16_384_in21ft1k',
    'deit3_large_patch16_384',
    'deit3_large_patch16_384_in21ft1k',
    'deit3_small_patch16_384',
    'deit3_small_patch16_384_in21ft1k',
    'deit_base_distilled_patch16_384',
    'deit_base_patch16_384',
    'eca_botnext26ts_256',
    'eca_halonext26ts',
    'halo2botnet50ts_256',
    'halonet26t',
    'halonet50ts',
    'halonet_h1',
    'lambda_resnet26rpt_256',
    'lamhalobotnet50ts_256',
    'maxvit_nano_rw_256',
    'maxvit_pico_rw_256',
    'maxvit_rmlp_nano_rw_256',
    'maxvit_rmlp_pico_rw_256',
    'maxvit_rmlp_small_rw_256',
    'maxvit_rmlp_tiny_rw_256',
    'maxvit_tiny_pm_256',
    'maxvit_tiny_rw_256',
    'maxxvit_nano_rw_256',
    'maxxvit_small_rw_256',
    'maxxvit_tiny_rw_256',
    'sebotnet33ts_256',
    'sehalonet33ts',
    'swin_base_patch4_window12_384',
    'swin_base_patch4_window12_384_in22k',
    'swin_large_patch4_window12_384',
    'swin_large_patch4_window12_384_in22k',
    'swinv2_base_window12_192_22k',
    'swinv2_base_window12to16_192to256_22kft1k',
    'swinv2_base_window12to24_192to384_22kft1k',
    'swinv2_base_window16_256',
    'swinv2_base_window8_256',
    'swinv2_cr_base_384',
    'swinv2_cr_giant_384',
    'swinv2_cr_huge_384',
    'swinv2_cr_large_384',
    'swinv2_cr_small_384',
    'swinv2_cr_tiny_384',
    'swinv2_large_window12_192_22k',
    'swinv2_large_window12to16_192to256_22kft1k',
    'swinv2_large_window12to24_192to384_22kft1k',
    'swinv2_small_window16_256',
    'swinv2_small_window8_256',
    'swinv2_tiny_window16_256',
    'swinv2_tiny_window8_256',
    'vit_base_patch16_384',
    'vit_base_patch16_plus_240',
    'vit_base_patch32_384',
    'vit_base_patch32_plus_256',
    'vit_base_r50_s16_384',
    'vit_base_resnet50_384',
    'vit_large_patch16_384',
    'vit_large_patch32_384',
    'vit_large_r50_s32_384',
    'vit_relpos_base_patch16_plus_240',
    'vit_relpos_base_patch32_plus_rpn_256',
    'vit_small_patch16_384',
    'vit_small_patch32_384',
    'vit_small_r26_s32_384',
    'vit_tiny_patch16_384',
    'vit_tiny_r_s16_p8_384',
    'volo_d1_384',
    'volo_d2_384',
    'volo_d3_448',
    'volo_d4_448',
    'volo_d5_448',
    'volo_d5_512',
]

MMPRETRAIN_BLACKLIST = [
    'barlowtwins_resnet50_8xb256-coslr-300e_in1k',
    'beit-g-p14_3rdparty-eva_30m',
    'beit-g-p14_eva-30m-in21k-pre_3rdparty_in1k-560px',
    'beit-g-p14_eva-30m-pre_3rdparty_in21k',
    'beit-g-p16_3rdparty-eva_30m',
    'beit-l-p14_3rdparty-eva_in21k',
    'beit-l-p14_eva-pre_3rdparty_in21k',
    'beit_beit-base-p16_8xb256-amp-coslr-300e_in1k',
    'beitv2_beit-base-p16_8xb256-amp-coslr-300e_in1k',
    'blip-base_3rdparty_caption',
    'blip-base_3rdparty_nlvr',
    'blip-base_3rdparty_retrieval',
    'blip-base_3rdparty_vqa',
    'blip-base_8xb16_refcoco',
    'blip2-opt2.7b_3rdparty-zeroshot_caption',
    'blip2-opt2.7b_3rdparty-zeroshot_vqa',
    'blip2_3rdparty_retrieval',
    'byol_resnet50_16xb256-coslr-200e_in1k',
    'cae_beit-base-p16_8xb256-amp-coslr-300e_in1k',
    'cn-clip_resnet50_zeroshot-cls_cifar100',
    'cn-clip_vit-base-p16_zeroshot-cls_cifar100',
    'cn-clip_vit-huge-p14_zeroshot-cls_cifar100',
    'cn-clip_vit-large-p14_zeroshot-cls_cifar100',
    'conformer-base-p16_3rdparty_in1k',
    'conformer-small-p16_3rdparty_in1k',
    'conformer-small-p32_8xb128_in1k',
    'conformer-tiny-p16_3rdparty_in1k',
    'convnext-base_3rdparty_in21k',
    'convnext-large_3rdparty_in21k',
    'convnext-xlarge_3rdparty_in21k',
    'densecl_resnet50_8xb32-coslr-200e_in1k',
    'efficientnetv2-l_3rdparty_in21k',
    'efficientnetv2-m_3rdparty_in21k',
    'efficientnetv2-s_3rdparty_in21k',
    'efficientnetv2-xl_3rdparty_in21k',
    'eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k',
    'flamingo_3rdparty-zeroshot_caption',
    'flamingo_3rdparty-zeroshot_vqa',
    'mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k',
    'mae_vit-base-p16_8xb512-amp-coslr-300e_in1k',
    'mae_vit-base-p16_8xb512-amp-coslr-400e_in1k',
    'mae_vit-base-p16_8xb512-amp-coslr-800e_in1k',
    'mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k',
    'mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k',
    'mae_vit-large-p16_8xb512-amp-coslr-400e_in1k',
    'mae_vit-large-p16_8xb512-amp-coslr-800e_in1k',
    'maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k',
    'milan_vit-base-p16_16xb256-amp-coslr-400e_in1k',
    'mixmim_mixmim-base_16xb128-coslr-300e_in1k',
    'mocov2_resnet50_8xb32-coslr-200e_in1k',
    'mocov3_resnet50_8xb512-amp-coslr-100e_in1k',
    'mocov3_resnet50_8xb512-amp-coslr-300e_in1k',
    'mocov3_resnet50_8xb512-amp-coslr-800e_in1k',
    'mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k',
    'mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k',
    'mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k',
    'ofa-base_3rdparty-finetuned_caption',
    'ofa-base_3rdparty-finetuned_refcoco',
    'ofa-base_3rdparty-finetuned_vqa',
    'ofa-base_3rdparty-zeroshot_vqa',
    'resnet101-csra_1xb16_voc07-448px',
    'resnet101_8xb16_cifar10',
    'resnet152_8xb16_cifar10',
    'resnet18_8xb16_cifar10',
    'resnet34_8xb16_cifar10',
    'resnet50-arcface_8xb32_inshop',
    'resnet50_8xb16_cifar10',
    'resnet50_8xb16_cifar100',
    'resnet50_8xb8_cub',
    'simclr_resnet50_16xb256-coslr-200e_in1k',
    'simclr_resnet50_16xb256-coslr-800e_in1k',
    'simmim_swin-base-w6_16xb128-amp-coslr-800e_in1k-192px',
    'simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px',
    'simmim_swin-large-w12_16xb128-amp-coslr-800e_in1k-192px',
    'simsiam_resnet50_8xb32-coslr-100e_in1k',
    'simsiam_resnet50_8xb32-coslr-200e_in1k',
    'swav_resnet50_8xb32-mcrop-coslr-200e_in1k-224px-96px',
    'swin-base_3rdparty_in1k-384',
    'swin-base_in21k-pre-3rdparty_in1k-384',
    'swin-l_glip-pre_3rdparty_384px',
    'swin-large_8xb8_cub-384px',
    'swin-large_in21k-pre-3rdparty_in1k-384',
    'swin-t_glip-pre_3rdparty',
    'swinv2-base-w12_3rdparty_in21k-192px',
    'swinv2-base-w16_3rdparty_in1k-256px',
    'swinv2-base-w16_in21k-pre_3rdparty_in1k-256px',
    'swinv2-base-w24_in21k-pre_3rdparty_in1k-384px',
    'swinv2-base-w8_3rdparty_in1k-256px',
    'swinv2-large-w12_3rdparty_in21k-192px',
    'swinv2-large-w16_in21k-pre_3rdparty_in1k-256px',
    'swinv2-large-w24_in21k-pre_3rdparty_in1k-384px',
    'swinv2-small-w16_3rdparty_in1k-256px',
    'swinv2-small-w8_3rdparty_in1k-256px',
    'swinv2-tiny-w16_3rdparty_in1k-256px',
    'swinv2-tiny-w8_3rdparty_in1k-256px',
    'vit-base-p14_dinov2-pre_3rdparty',
    'vit-base-p14_eva02-pre_in21k',
    'vit-base-p16_sam-pre_3rdparty_sa1b-1024px',
    'vit-giant-p14_dinov2-pre_3rdparty',
    'vit-huge-p14_mae-1600e-pre_32xb8-coslr-50e_in1k-448px',
    'vit-huge-p16_sam-pre_3rdparty_sa1b-1024px',
    'vit-large-p14_clip-openai-pre_3rdparty',
    'vit-large-p14_dinov2-pre_3rdparty',
    'vit-large-p14_eva02-pre_in21k',
    'vit-large-p14_eva02-pre_m38m',
    'vit-large-p16_sam-pre_3rdparty_sa1b-1024px',
    'vit-small-p14_dinov2-pre_3rdparty',
    'vit-small-p14_eva02-pre_in21k',
    'vit-tiny-p14_eva02-pre_in21k',
]


PYTORCHCV_HAS_CHECKPOINT = [
    'airnet50_1x64d_r16_pytorchcv',
    'airnet50_1x64d_r2_pytorchcv',
    'airnext50_32x4d_r2_pytorchcv',
    'alexnet_pytorchcv',
    'alexnetb_pytorchcv',
    'bagnet17_pytorchcv',
    'bagnet33_pytorchcv',
    'bagnet9_pytorchcv',
    'bam_resnet50_pytorchcv',
    'bn_vgg11_pytorchcv',
    'bn_vgg11b_pytorchcv',
    'bn_vgg13_pytorchcv',
    'bn_vgg13b_pytorchcv',
    'bn_vgg16_pytorchcv',
    'bn_vgg16b_pytorchcv',
    'bn_vgg19_pytorchcv',
    'bn_vgg19b_pytorchcv',
    'bninception_pytorchcv',
    'cbam_resnet50_pytorchcv',
    'condensenet74_c4_g4_pytorchcv',
    'condensenet74_c8_g8_pytorchcv',
    'darknet53_pytorchcv',
    'darknet_ref_pytorchcv',
    'darknet_tiny_pytorchcv',
    'darts_pytorchcv',
    'densenet121_pytorchcv',
    'densenet161_pytorchcv',
    'densenet169_pytorchcv',
    'densenet201_pytorchcv',
    'dicenet_w1_pytorchcv',
    'dicenet_w2_pytorchcv',
    'dicenet_w3d2_pytorchcv',
    'dicenet_w3d4_pytorchcv',
    'dicenet_w5d4_pytorchcv',
    'dicenet_w7d8_pytorchcv',
    'dicenet_wd2_pytorchcv',
    'dicenet_wd5_pytorchcv',
    'diracnet18v2_pytorchcv',
    'diracnet34v2_pytorchcv',
    'dla102_pytorchcv',
    'dla102x2_pytorchcv',
    'dla102x_pytorchcv',
    'dla169_pytorchcv',
    'dla34_pytorchcv',
    'dla46c_pytorchcv',
    'dla46xc_pytorchcv',
    'dla60_pytorchcv',
    'dla60x_pytorchcv',
    'dla60xc_pytorchcv',
    'dpn131_pytorchcv',
    'dpn68_pytorchcv',
    'dpn98_pytorchcv',
    'drnc26_pytorchcv',
    'drnc42_pytorchcv',
    'drnc58_pytorchcv',
    'drnd105_pytorchcv',
    'drnd22_pytorchcv',
    'drnd38_pytorchcv',
    'drnd54_pytorchcv',
    'efficientnet_b0_pytorchcv',
    'efficientnet_b0b_pytorchcv',
    'efficientnet_b0c_pytorchcv',
    'efficientnet_b1_pytorchcv',
    'efficientnet_b1b_pytorchcv',
    'efficientnet_b1c_pytorchcv',
    'efficientnet_b2b_pytorchcv',
    'efficientnet_b2c_pytorchcv',
    'efficientnet_b3b_pytorchcv',
    'efficientnet_b3c_pytorchcv',
    'efficientnet_b4b_pytorchcv',
    'efficientnet_b4c_pytorchcv',
    'efficientnet_b5b_pytorchcv',
    'efficientnet_b5c_pytorchcv',
    'efficientnet_b6b_pytorchcv',
    'efficientnet_b6c_pytorchcv',
    'efficientnet_b7b_pytorchcv',
    'efficientnet_b7c_pytorchcv',
    'efficientnet_b8c_pytorchcv',
    'efficientnet_edge_large_b_pytorchcv',
    'efficientnet_edge_medium_b_pytorchcv',
    'efficientnet_edge_small_b_pytorchcv',
    'espnetv2_w1_pytorchcv',
    'espnetv2_w2_pytorchcv',
    'espnetv2_w3d2_pytorchcv',
    'espnetv2_w5d4_pytorchcv',
    'espnetv2_wd2_pytorchcv',
    'fbnet_cb_pytorchcv',
    'fdmobilenet_w1_pytorchcv',
    'fdmobilenet_w3d4_pytorchcv',
    'fdmobilenet_wd2_pytorchcv',
    'fdmobilenet_wd4_pytorchcv',
    'fishnet150_pytorchcv',
    'hardnet39ds_pytorchcv',
    'hardnet68_pytorchcv',
    'hardnet68ds_pytorchcv',
    'hardnet85_pytorchcv',
    'hrnet_w18_small_v1_pytorchcv',
    'hrnet_w18_small_v2_pytorchcv',
    'hrnetv2_w18_pytorchcv',
    'hrnetv2_w30_pytorchcv',
    'hrnetv2_w32_pytorchcv',
    'hrnetv2_w40_pytorchcv',
    'hrnetv2_w44_pytorchcv',
    'hrnetv2_w48_pytorchcv',
    'hrnetv2_w64_pytorchcv',
    'ibn_densenet121_pytorchcv',
    'ibn_densenet169_pytorchcv',
    'ibn_resnet101_pytorchcv',
    'ibn_resnet50_pytorchcv',
    'ibn_resnext101_32x4d_pytorchcv',
    'ibnb_resnet50_pytorchcv',
    'igcv3_w1_pytorchcv',
    'igcv3_w3d4_pytorchcv',
    'igcv3_wd2_pytorchcv',
    'igcv3_wd4_pytorchcv',
    'irevnet301_pytorchcv',
    'menet108_8x1_g3_pytorchcv',
    'menet128_8x1_g4_pytorchcv',
    'menet160_8x1_g8_pytorchcv',
    'menet228_12x1_g3_pytorchcv',
    'menet256_12x1_g4_pytorchcv',
    'menet348_12x1_g3_pytorchcv',
    'menet352_12x1_g8_pytorchcv',
    'menet456_24x1_g3_pytorchcv',
    'mixnet_l_pytorchcv',
    'mixnet_m_pytorchcv',
    'mixnet_s_pytorchcv',
    'mnasnet_a1_pytorchcv',
    'mnasnet_b1_pytorchcv',
    'mobilenet_w1_pytorchcv',
    'mobilenet_w3d4_pytorchcv',
    'mobilenet_wd2_pytorchcv',
    'mobilenet_wd4_pytorchcv',
    'mobilenetb_w1_pytorchcv',
    'mobilenetb_w3d4_pytorchcv',
    'mobilenetb_wd2_pytorchcv',
    'mobilenetb_wd4_pytorchcv',
    'mobilenetv2_w1_pytorchcv',
    'mobilenetv2_w3d4_pytorchcv',
    'mobilenetv2_wd2_pytorchcv',
    'mobilenetv2_wd4_pytorchcv',
    'mobilenetv2b_w1_pytorchcv',
    'mobilenetv2b_w3d4_pytorchcv',
    'mobilenetv2b_wd2_pytorchcv',
    'mobilenetv2b_wd4_pytorchcv',
    'mobilenetv3_large_w1_pytorchcv',
    'nasnet_4a1056_pytorchcv',
    'peleenet_pytorchcv',
    'preresnet101_pytorchcv',
    'preresnet101b_pytorchcv',
    'preresnet10_pytorchcv',
    'preresnet12_pytorchcv',
    'preresnet14_pytorchcv',
    'preresnet152_pytorchcv',
    'preresnet152b_pytorchcv',
    'preresnet16_pytorchcv',
    'preresnet18_pytorchcv',
    'preresnet18_w3d4_pytorchcv',
    'preresnet18_wd2_pytorchcv',
    'preresnet18_wd4_pytorchcv',
    'preresnet200b_pytorchcv',
    'preresnet269b_pytorchcv',
    'preresnet26_pytorchcv',
    'preresnet34_pytorchcv',
    'preresnet50_pytorchcv',
    'preresnet50b_pytorchcv',
    'preresnetbc14b_pytorchcv',
    'preresnetbc26b_pytorchcv',
    'preresnetbc38b_pytorchcv',
    'proxylessnas_cpu_pytorchcv',
    'proxylessnas_gpu_pytorchcv',
    'proxylessnas_mobile14_pytorchcv',
    'proxylessnas_mobile_pytorchcv',
    'pyramidnet101_a360_pytorchcv',
    'resnesta101_pytorchcv',
    'resnesta152_pytorchcv',
    'resnesta18_pytorchcv',
    'resnesta200_pytorchcv',
    'resnesta269_pytorchcv',
    'resnesta50_pytorchcv',
    'resnestabc14_pytorchcv',
    'resnestabc26_pytorchcv',
    'resnet101_pytorchcv',
    'resnet101b_pytorchcv',
    'resnet10_pytorchcv',
    'resnet12_pytorchcv',
    'resnet14_pytorchcv',
    'resnet152_pytorchcv',
    'resnet152b_pytorchcv',
    'resnet16_pytorchcv',
    'resnet18_pytorchcv',
    'resnet18_w3d4_pytorchcv',
    'resnet18_wd2_pytorchcv',
    'resnet18_wd4_pytorchcv',
    'resnet26_pytorchcv',
    'resnet34_pytorchcv',
    'resnet50_pytorchcv',
    'resnet50b_pytorchcv',
    'resneta101b_pytorchcv',
    'resneta10_pytorchcv',
    'resneta152b_pytorchcv',
    'resneta18_pytorchcv',
    'resneta50b_pytorchcv',
    'resnetabc14b_pytorchcv',
    'resnetbc14b_pytorchcv',
    'resnetbc26b_pytorchcv',
    'resnetbc38b_pytorchcv',
    'resnetd101b_pytorchcv',
    'resnetd152b_pytorchcv',
    'resnetd50b_pytorchcv',
    'resnext101_32x4d_pytorchcv',
    'resnext101_64x4d_pytorchcv',
    'resnext14_16x4d_pytorchcv',
    'resnext14_32x2d_pytorchcv',
    'resnext14_32x4d_pytorchcv',
    'resnext26_32x2d_pytorchcv',
    'resnext26_32x4d_pytorchcv',
    'resnext50_32x4d_pytorchcv',
    'scnet101_pytorchcv',
    'scnet50_pytorchcv',
    'scneta50_pytorchcv',
    'selecsls42b_pytorchcv',
    'selecsls60_pytorchcv',
    'selecsls60b_pytorchcv',
    'senet154_pytorchcv',
    'senet16_pytorchcv',
    'senet28_pytorchcv',
    'sepreresnet10_pytorchcv',
    'sepreresnet12_pytorchcv',
    'sepreresnet16_pytorchcv',
    'sepreresnet18_pytorchcv',
    'sepreresnet26_pytorchcv',
    'sepreresnet50b_pytorchcv',
    'sepreresnetbc26b_pytorchcv',
    'sepreresnetbc38b_pytorchcv',
    'seresnet101_pytorchcv',
    'seresnet101b_pytorchcv',
    'seresnet10_pytorchcv',
    'seresnet12_pytorchcv',
    'seresnet14_pytorchcv',
    'seresnet152_pytorchcv',
    'seresnet16_pytorchcv',
    'seresnet18_pytorchcv',
    'seresnet26_pytorchcv',
    'seresnet50_pytorchcv',
    'seresnet50b_pytorchcv',
    'seresnetbc26b_pytorchcv',
    'seresnetbc38b_pytorchcv',
    'seresnext101_32x4d_pytorchcv',
    'seresnext101_64x4d_pytorchcv',
    'seresnext50_32x4d_pytorchcv',
    'shufflenet_g1_w1_pytorchcv',
    'shufflenet_g1_w3d4_pytorchcv',
    'shufflenet_g1_wd2_pytorchcv',
    'shufflenet_g1_wd4_pytorchcv',
    'shufflenet_g2_w1_pytorchcv',
    'shufflenet_g3_w1_pytorchcv',
    'shufflenet_g3_w3d4_pytorchcv',
    'shufflenet_g3_wd2_pytorchcv',
    'shufflenet_g3_wd4_pytorchcv',
    'shufflenet_g4_w1_pytorchcv',
    'shufflenet_g8_w1_pytorchcv',
    'shufflenetv2_w1_pytorchcv',
    'shufflenetv2_w2_pytorchcv',
    'shufflenetv2_w3d2_pytorchcv',
    'shufflenetv2_wd2_pytorchcv',
    'shufflenetv2b_w1_pytorchcv',
    'shufflenetv2b_w2_pytorchcv',
    'shufflenetv2b_w3d2_pytorchcv',
    'shufflenetv2b_wd2_pytorchcv',
    'spnasnet_pytorchcv',
    'sqnxt23_w1_pytorchcv',
    'sqnxt23_w2_pytorchcv',
    'sqnxt23_w3d2_pytorchcv',
    'sqnxt23v5_w1_pytorchcv',
    'sqnxt23v5_w2_pytorchcv',
    'sqnxt23v5_w3d2_pytorchcv',
    'squeezenet_v1_0_pytorchcv',
    'squeezenet_v1_1_pytorchcv',
    'squeezeresnet_v1_0_pytorchcv',
    'squeezeresnet_v1_1_pytorchcv',
    'vgg11_pytorchcv',
    'vgg13_pytorchcv',
    'vgg16_pytorchcv',
    'vgg19_pytorchcv',
    'vovnet27s_pytorchcv',
    'vovnet39_pytorchcv',
    'vovnet57_pytorchcv',
    'wrn50_2_pytorchcv',
    'zfnet_pytorchcv',
    'zfnetb_pytorchcv'
]
