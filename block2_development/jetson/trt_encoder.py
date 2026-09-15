"""TensorRT encoder for block 2 v2 - a drop-in for `aster_block2.models.Encoder`.

Same loading logic as the deployed `aster_pipeline/tensorrt_mil.py` (static-batch engine,
execute_async_v3 on the current torch stream), wrapped as an nn.Module so that
`SessionScorer` calls it exactly as it calls the PyTorch encoder: any number of crops in,
[N, 512] float32 features out. The last partial chunk is zero-padded to the static batch
and the padding rows are dropped.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

_TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16,
                 trt.int32: torch.int32, trt.int8: torch.int8, trt.bool: torch.bool}


class TensorRTEncoder(nn.Module):
    def __init__(self, engine_path: Path) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("the TensorRT encoder needs CUDA")
        # The runtime must outlive the engine and the context: keep it on the instance.
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path} - was it built on THIS Jetson?")
        self._context = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        mode = self._engine.get_tensor_mode
        self._input = next(n for n in names if mode(n) == trt.TensorIOMode.INPUT)
        self._output = next(n for n in names if mode(n) == trt.TensorIOMode.OUTPUT)
        self.input_shape = tuple(self._engine.get_tensor_shape(self._input))
        self.output_shape = tuple(self._engine.get_tensor_shape(self._output))
        if -1 in self.input_shape:
            raise ValueError(f"expected a static-batch engine, got {self.input_shape}")
        self.batch = self.input_shape[0]
        self._input_dtype = _TRT_TO_TORCH[self._engine.get_tensor_dtype(self._input)]
        self._output_dtype = _TRT_TO_TORCH[self._engine.get_tensor_dtype(self._output)]

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        stream = torch.cuda.current_stream()
        for offset in range(0, len(images), self.batch):
            chunk = images[offset:offset + self.batch]
            valid = len(chunk)
            if valid < self.batch:
                pad = torch.zeros(self.batch - valid, *chunk.shape[1:], dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad])
            chunk = chunk.to("cuda", dtype=self._input_dtype).contiguous()
            output = torch.empty(self.output_shape, device="cuda", dtype=self._output_dtype)
            self._context.set_tensor_address(self._input, chunk.data_ptr())
            self._context.set_tensor_address(self._output, output.data_ptr())
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT encoder execution failed")
            outputs.append(output[:valid].float())
        stream.synchronize()
        return torch.cat(outputs)
