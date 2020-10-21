# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .build import make_optimizer, make_cdb_optimizer
from .build import make_lr_scheduler, make_lr_cdb_scheduler
from .lr_scheduler import WarmupMultiStepLR
