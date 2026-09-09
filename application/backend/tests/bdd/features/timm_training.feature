# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Timm Model Training Smoke Test
  As a maintainer of the application
  I want to train one representative model per unique timm architecture flavor
  So that I catch runtime failures across unmanaged third-party model families early

  Background:
    Given a project "flowers-smoke" is created from the dataset archive at "https://storage.geti.intel.com/test-data/geti/datasets/qa/flowers-geti-medium.zip"

  @timm
  Scenario Outline: Train a representative model for each unique timm architecture flavor
    Given the training configuration for model architecture "<model_architecture_id>" is set to 10 epochs
    When I train timm model architecture "<model_architecture_id>" on device "cpu"
    Then the trained model has a "openvino" variant with a positive weights size

    # The following examples table contains a representative model architecture for each unique timm architecture flavor.
    # The list of model architectures was generated with `just list-timm-smoke-models --format gherkin`.
    # gemma4_vit and mobilenetv5 removed from the scope because they have 167.4 and 294.1 million parameters respectively,
      # which is too large for smoke testing.
    Examples: Timm architecture flavors
      | family                    | model_architecture_id                     |
      | beit                      | beit_base_patch16_224.in22k_ft_in22k      |
      | byoanet                   | haloregnetz_b.ra3_in1k                    |
      | byobnet                   | bat_resnext26ts.ch_in1k                   |
      | byobnet                   | test_byobnet.r160_in1k                    |
      | cait                      | cait_xxs24_224.fb_dist_in1k               |
      | coat                      | coat_lite_tiny.in1k                       |
      | convit                    | convit_tiny.fb_in1k                       |
      | convmixer                 | convmixer_1024_20_ks9_p14.in1k            |
      | convnext                  | test_convnext.r160_in1k                   |
      | crossvit                  | crossvit_tiny_240.in1k                    |
      | csatv2                    | csatv2.r512_in1k                          |
      | cspnet                    | cs3darknet_focus_s.ra4_e3600_r256_in1k    |
      | davit                     | davit_tiny.msft_in1k                      |
      | deit                      | deit_tiny_patch16_224.fb_in1k             |
      | densenet                  | densenet121.ra_in1k                       |
      | dla                       | dla46x_c.in1k                             |
      | dpn                       | dpn68.mx_in1k                             |
      | edgenext                  | edgenext_xx_small.in1k                    |
      | efficientformer           | efficientformer_l1.snap_dist_in1k         |
      | efficientformer_v2        | efficientformerv2_s0.snap_dist_in1k       |
      | efficientnet              | tinynet_e.in1k                            |
      | efficientvit_mit          | efficientvit_b0.r224_in1k                 |
      | efficientvit_msra         | efficientvit_m0.r224_in1k                 |
      | eva                       | eva02_tiny_patch14_224.mim_in22k          |
      | fasternet                 | fasternet_t0.in1k                         |
      | focalnet                  | focalnet_tiny_srf.ms_in1k                 |
      | gcvit                     | gcvit_xxtiny.in1k                         |
      | ghostnet                  | ghostnet_100.in1k                         |
      | hardcorenas               | hardcorenas_a.miil_green_in1k             |
      | hgnet                     | hgnetv2_b0.ssld_stage1_in22k_in1k         |
      | hiera                     | hiera_tiny_224.mae                        |
      | hieradet_sam2             | sam2_hiera_tiny.fb_r896                   |
      | hrnet                     | hrnet_w18_small.gluon_in1k                |
      | inception_next            | inception_next_atto.sail_in1k             |
      | inception_resnet_v2       | inception_resnet_v2.tf_ens_adv_in1k       |
      | inception_v3              | inception_v3.gluon_in1k                   |
      | inception_v4              | inception_v4.tf_in1k                      |
      | levit                     | levit_128s.fb_dist_in1k                   |
      | mambaout                  | mambaout_femto.in1k                       |
      | maxxvit                   | maxvit_rmlp_pico_rw_256.sw_in1k           |
      | metaformer                | poolformer_s12.sail_in1k                  |
      | mlp_mixer                 | resmlp_12_224.fb_dino                     |
      | mobilenetv3               | mobilenetv3_small_050.lamb_in1k           |
      | mvitv2                    | mvitv2_tiny.fb_in1k                       |
      | naflexvit                 | naflexvit_base_patch16_gap.e300_s576_in1k |
      | nasnet                    | nasnetalarge.tf_in1k                      |
      | nest                      | nest_tiny_jx.goog_in1k                    |
      | nextvit                   | nextvit_small.bd_in1k                     |
      | nfnet                     | test_nfnet.r160_in1k                      |
      | pit                       | pit_ti_224.in1k                           |
      | pnasnet                   | pnasnet5large.tf_in1k                     |
      | pvt_v2                    | pvt_v2_b0.in1k                            |
      | rdnet                     | rdnet_tiny.nv_in1k                        |
      | regnet                    | regnetx_002.pycls_in1k                    |
      | repghost                  | repghostnet_050.in1k                      |
      | repvit                    | repvit_m0_9.dist_300e_in1k                |
      | resnest                   | resnest14d.gluon_in1k                     |
      | resnet                    | test_resnet.r160_in1k                     |
      | resnetv2                  | resnetv2_18.ra4_e3600_r224_in1k           |
      | rexnet                    | rexnet_100.nav_in1k                       |
      | sequencer                 | sequencer2d_s.in1k                        |
      | shvit                     | shvit_s1.in1k                             |
      | sknet                     | skresnet18.ra_in1k                        |
      | starnet                   | starnet_s1.in1k                           |
      | swiftformer               | swiftformer_xs.dist_in1k                  |
      | swin_transformer          | swin_tiny_patch4_window7_224.ms_in1k      |
      | swin_transformer_v2       | swinv2_tiny_window8_256.ms_in1k           |
      | swin_transformer_v2_cr    | swinv2_cr_tiny_ns_224.sw_in1k             |
      | tiny_vit                  | tiny_vit_5m_224.dist_in22k                |
      | tnt                       | tnt_s_legacy_patch16_224.in1k             |
      | tresnet                   | tresnet_m.miil_in1k                       |
      | twins                     | twins_svt_small.in1k                      |
      | vgg                       | vgg11.tv_in1k                             |
      | visformer                 | visformer_tiny.in1k                       |
      | vision_transformer        | test_vit.r160_in1k                        |
      | vision_transformer_hybrid | vit_tiny_r_s16_p8_224.augreg_in21k        |
      | vision_transformer_relpos | vit_srelpos_small_patch16_224.sw_in1k     |
      | vision_transformer_sam    | samvit_base_patch16.sa1b                  |
      | vitamin                   | vitamin_small_224.datacomp1b_clip         |
      | volo                      | volo_d1_224.sail_in1k                     |
      | vovnet                    | ese_vovnet19b_dw.ra_in1k                  |
      | xception_aligned          | xception41.tf_in1k                        |
      | xcit                      | xcit_nano_12_p16_224.fb_dist_in1k         |
