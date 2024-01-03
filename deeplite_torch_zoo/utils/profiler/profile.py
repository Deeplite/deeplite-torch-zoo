import warnings

from pandas import DataFrame

from deeplite_torch_zoo.utils.profiler.handlers import handlers
from deeplite_torch_zoo.utils.profiler.utils.trace import trace
from deeplite_torch_zoo.utils.profiler.utils.placer import Placer, get_nodes


def profile_macs(model, args=(), kwargs=None, reduction=sum):
    results = {}

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


def profile_ram(model, args=(), kwargs=None, num_bytes=4, detailed=False):
    graph = trace(model, args, kwargs)
    nodes = get_nodes(graph)
    placer = Placer(nodes)
    nodes = placer.place(num_bytes=num_bytes)

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
            'ram',
            'scope',
        ],
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
        df.scope[node.name] = node.scope

    if detailed:
        return df
    else:
        return df.ram.max() / 2**20


def ram_report(df, topk='top1', verbose=False, export=False, filename="report"):
    if verbose:
        idx_max = df.index[df.ram == df.ram.max()]
        print('-' * 120)
        print(df.to_string())
        print('-' * 120)
        print(f' >> Peak Memory of {df.ram.max() / (2**10):.0f} kB found in the following node(s):')
        if topk == 'top1':
            for idx in idx_max:
                print(df.loc[idx].to_string())
                print()
    if export:
        export_path = f"{filename}.csv"
        df.to_csv(export_path)
        print(f"RAM usage report exported to {export_path}")

