base = dict(
    find_unused_parameters=True,
    nncf_config = dict(
        input_info=dict(sample_size=[1, 3, 256, 256]),
        compression=[],
        log_dir='.')
)
int8 = dict(
    optimizer = dict(lr=0.00025),
    total_epochs = 2,
    nncf_config = dict(
        compression=[
            dict(
                algorithm='quantization',
                initializer=dict(
                    range=dict(num_init_steps=10),
                    batchnorm_adaptation=dict(num_bn_adaptation_steps=30))),
            ],
        )
)
sparsity = dict(
    optimizer = dict(lr=0.00025),
    total_epochs = 50,
    nncf_config = dict(
        compression=[
            dict(
                algorithm='magnitude_sparsity',
                params=dict(
                    schedule='multistep',
                    multistep_sparsity_levels=[0.3, 0.5, 0.7],
                    multistep_steps=[40, 80]))
            ],
        )
)

order_of_parts = ['int8', 'sparsity']
