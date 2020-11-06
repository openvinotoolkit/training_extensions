exp_num = 0

dataset = 'celeba_spoof'

datasets = dict(LCCFASD_root='/home/prokofiev/pytorch/LCC_FASDnew',
                Celeba_root='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof',
                Casia_root='/home/prokofiev/pytorch/antispoofing/CASIA')

external = dict(train_params=dict(), val_params=dict(), test_params=dict())

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229], std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,50], gamma=0.2)

data = dict(cuda=True, batch_size=256, data_loader_workers=4,
            sampler=None, pin_memory=True, data_root='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof')

resize = dict(height=128, width=128)

checkpoint = dict(snapshot_name="MobileNet3_100.pth.tar",
                  experiment_path='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_Celeba_100')

loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5, s=1, margin_type='cross_entropy', label_smooth=False, smoothing=0.1, ratio=[1,1], gamma=0),
            soft_triple=dict(cN=2, K=10, s=1, tau=.2, m=0.35))

epochs = dict(start_epoch=0, max_epoch=71)

model= dict(model_type='Mobilenet3', model_size='large',
            pretrained=True, embeding_dim=1280, width_mult=.75,
            imagenet_weights='/home/prokofiev/pytorch/antispoofing/weights/mobilenetv3-large-0.75-9632d2a8.pth',
            to_forward=False)

aug = dict(type_aug=None, alpha=0.5, beta=0.5, cutmix_prob=0.7)

curves = dict(det_curve='det_curve_35.png', roc_curve='roc_curve_35.png')

dropout = dict(prob_dropout=0.1, classifier=0.35, type='bernoulli', mu=0.5, sigma=0.3)

data_parallel = dict(use_parallel=False, parallel_params=dict(device_ids=[0,1], output_device=0))

evaulation = True

RSC = dict(use_rsc=False, p=0.333, b=0.333)

multi_task_learning = True

test_dataset = dict(type='celeba_spoof')

conv_cd = dict(theta=0.7)

test_steps = 10
