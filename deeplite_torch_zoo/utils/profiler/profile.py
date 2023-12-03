import warnings

from pandas import DataFrame

from .handlers import handlers
from .utils.trace import trace
from .utils.placer import Placer, get_nodes


__all__ = ['profile_macs', 'profile_ram']


def profile_macs(model, args=(), kwargs=None, reduction=sum):
    results = dict()

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break
        else:
            warnings.warn(f'No handlers found: "{node.operator}". Skipped.')

    if reduction is not None:
        return reduction(results.values())

    return results


def profile_ram(model, args=(), kwargs=None):
    graph = trace(model, args, kwargs)
    nodes = get_nodes(graph)
    placer = Placer(nodes)
    nodes = placer.place(num_bytes=4)

    df = DataFrame(
        index=[node.name for node in nodes],
        columns=[
            'weight',
            'bias',
            'input_shape',
            'output_shape',
            'in_tensors',
            'out_tensors',
            'active_blocks',
            'ram'
        ]
    )

    for node in nodes:
        df.weight[node.name] = node.weights
        df.bias[node.name] = node.bias
        df.input_shape[node.name] = [x.shape for x in node.inputs]
        df.output_shape[node.name] = [x.shape for x in node.outputs]
        df.in_tensors[node.name] = [x.name for x in node.inputs]
        df.out_tensors[node.name] = [x.name for x in node.outputs]
        df.active_blocks[node.name] = node.malloc_blocks
        df.ram[node.name] = node.malloc_val

    return df.ram.max() / 2 ** 20
