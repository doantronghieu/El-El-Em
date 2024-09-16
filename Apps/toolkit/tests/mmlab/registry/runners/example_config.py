# Project configuration
project_name = 'simple_project'
work_dir = './work_dir'
seed = 42

# Model configuration
model = dict(
    type='SimpleModel'
)

# DataLoader configuration
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='SimpleDataset',
        num_samples=1000
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='SimpleDataset',
        num_samples=200
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=5,
    val_interval=1
)

val_cfg = dict()

# Evaluation configuration
val_evaluator = dict(type='SimpleMSE', prefix='val')

# Optimizer configuration
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01
    )
)

# Learning rate scheduler configuration
param_schedulers = dict(
    type='MultiStepLR',
    milestones=[2, 4],
    gamma=0.1
)

# Runner configuration
runner = dict(
    type='EpochBasedRunner',
    max_epochs=5,
    val_interval=1,
    log_interval=50
)

cfg = dict()

# Hooks configuration
hooks = [
    dict(type='IterTimerHook'),
    dict(type='LoggerHook', interval=50),
    dict(type='ParamSchedulerHook'),
    dict(type='CheckpointHook', priority=None, interval=1),
    dict(type='DistSamplerSeedHook')
]

# Default hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Feature flags configuration
feature_flags = dict()

# Checkpoint configuration
checkpoint = dict(
    interval=1
)

# Visualizer configuration
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend')
    ]
)

# Resume and load from configuration
resume_from = None
load_from = None

# CUDA configuration
cudnn_benchmark = False
mp_start_method = 'fork'

# Distributed configuration
dist_params = dict(
    backend='nccl'
)

# Log configurations
log = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
log_level = 'INFO'
default_scope = 'mmengine'
log_processor = dict(
    window_size=10,
    by_epoch=True,
    custom_cfg=None,
    num_digits=4
)


# Launcher configuration
launcher = 'none'
distributed = False

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0
    ),
    dist_cfg=dict(
        backend='nccl'
    )
)
