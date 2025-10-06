"""
Contains implementations of various optimization strategies.
"""
from tvm import te
from .base import OptimizationStrategy

class BaselineStrategy(OptimizationStrategy):
    """A baseline strategy with no optimizations."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        pass

class ParallelStrategy(OptimizationStrategy):
    """Applies parallelization to the output channel axis."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        _, oc, _, _ = operator.op.axis
        schedule[operator.op].parallel(oc)

class VectorizeStrategy(OptimizationStrategy):
    """Applies vectorization to the inner-most spatial axis."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        vector_width = self.params.get("vector_width", 8)
        _, _, _, ow = operator.op.axis
        wo, wi = schedule[operator.op].split(ow, factor=vector_width)
        schedule[operator.op].vectorize(wi)

class TilingStrategy(OptimizationStrategy):
    """Applies tiling and reordering to spatial and channel axes."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        tile_oc = self.params.get("tile_oc", 8)
        tile_ow = self.params.get("tile_ow", 8)

        n, oc, oh, ow = operator.op.axis
        ic, kh, kw = operator.op.reduce_axis

        oco, oci = schedule[operator.op].split(oc, factor=tile_oc)
        owo, owi = schedule[operator.op].split(ow, factor=tile_ow)

        schedule[operator.op].reorder(n, oco, oh, owo, oci, ic, kh, kw, owi)

class TilingAndParallelStrategy(TilingStrategy):
    """Combines Tiling and Parallelization."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        tile_oc = self.params.get("tile_oc", 8)
        
        n, oc, oh, ow = operator.op.axis
        oco, oci = schedule[operator.op].split(oc, factor=tile_oc)
        
        # Apply parallelization on the outer channel loop
        schedule[operator.op].parallel(oco)

class TilingAndVectorizeStrategy(TilingStrategy):
    """Combines Tiling and Vectorization."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        vector_width = self.params.get("vector_width", 8)
        tile_ow = self.params.get("tile_ow", 8)
        
        n, oc, oh, ow = operator.op.axis
        owo, owi = schedule[operator.op].split(ow, factor=tile_ow)
        
        # Apply vectorization on the inner-most width loop
        schedule[operator.op].vectorize(owi)

class AllCombinedStrategy(OptimizationStrategy):
    """Applies Tiling, Reordering, Parallelization, and Vectorization."""
    def apply(self, schedule: te.Schedule, operator: te.Tensor) -> None:
        vector_width = self.params.get("vector_width", 8)
        tile_oc = self.params.get("tile_oc", 8)
        tile_ow = self.params.get("tile_ow", 8)
        tile_ic = self.params.get("tile_ic", 4)
        
        n, oc, oh, ow = operator.op.axis
        ic, kh, kw = operator.op.reduce_axis

        # Tiling
        oco, oci = schedule[operator.op].split(oc, factor=tile_oc)
        owo, owi = schedule[operator.op].split(ow, factor=tile_ow)
        ico, ici = schedule[operator.op].split(ic, factor=tile_ic)

        # Reordering
        schedule[operator.op].reorder(n, oco, oh, owo, ico, kh, kw, oci, owi, ici)

        # Parallelization
        schedule[operator.op].parallel(oco)

        # Vectorization
        schedule[operator.op].vectorize(owi)
