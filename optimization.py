"""
"""

from typing import Any
from typing import Callable
from typing import ParamSpec
import spaces
import torch
from spaces.zero.torch.aoti import ZeroGPUCompiledModel
from spaces.zero.torch.aoti import ZeroGPUWeights
from torch.utils._pytree import tree_map

P = ParamSpec('P')

TRANSFORMER_IMAGE_DIM = torch.export.Dim('image_seq_length', min=4096, max=16384) # min: 0 images, max: 3 (1024x1024) images

TRANSFORMER_DYNAMIC_SHAPES = {
    'double': {
        'hidden_states': {
            1: TRANSFORMER_IMAGE_DIM,
        },
        'image_rotary_emb': (
            {0: TRANSFORMER_IMAGE_DIM + 512},
            {0: TRANSFORMER_IMAGE_DIM + 512},
        ),
    },
    'single': {
        'hidden_states': {
            1: TRANSFORMER_IMAGE_DIM + 512,
        },
        'image_rotary_emb': (
            {0: TRANSFORMER_IMAGE_DIM + 512},
            {0: TRANSFORMER_IMAGE_DIM + 512},
        ),
    },
}

INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}

def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):

    blocks = {
        'double': pipeline.transformer.transformer_blocks,
        'single': pipeline.transformer.single_transformer_blocks,
    }

    @spaces.GPU(duration=1200)
    def compile_block(blocks_kind: str):
        block = blocks[blocks_kind][0]
        with spaces.aoti_capture(block) as call:
            pipeline(*args, **kwargs)

        dynamic_shapes = tree_map(lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES[blocks_kind]

        with torch.no_grad():      
            exported = torch.export.export(
                mod=block,
                args=call.args,
                kwargs=call.kwargs,
                dynamic_shapes=dynamic_shapes,
            )

        return spaces.aoti_compile(exported, INDUCTOR_CONFIGS).archive_file

    for blocks_kind in ('double', 'single'):
        archive_file = compile_block(blocks_kind)
        for block in blocks[blocks_kind]:
            weights = ZeroGPUWeights(block.state_dict())
            block.forward = ZeroGPUCompiledModel(archive_file, weights)
