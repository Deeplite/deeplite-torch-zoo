import re
import warnings

import torch
import torch.jit

from deeplite_torch_zoo.utils.profiler.utils.flatten import Flatten
from deeplite_torch_zoo.utils.profiler.utils.ir import Graph, Node, Variable


class ScopeNameContextManager:
    """
    A context manager to handle scope names in PyTorch model tracing.
    This class temporarily overrides the '_slow_forward' method of torch.nn.Module
    to capture scope names accurately during the tracing process.
    """

    def __init__(self):
        self.original_slow_forward = None

    def __enter__(self):
        self.original_slow_forward = torch.nn.Module._slow_forward
        torch.nn.Module._slow_forward = self._patched_slow_forward

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.Module._slow_forward = self.original_slow_forward

    @staticmethod
    def _patched_slow_forward(module, *inputs, **kwargs):
        tracing_state = torch._C._get_tracing_state()

        if not tracing_state or isinstance(module.forward, torch._C.ScriptMethod):
            return module.forward(*inputs, **kwargs)

        if not hasattr(tracing_state, '_traced_module_stack'):
            tracing_state._traced_module_stack = []

        module_name = ScopeNameContextManager._get_tracing_name(module, tracing_state)
        scope_name = f'{module._get_name()}[{module_name}]' if module_name else module._get_name()
        tracing_state.push_scope(scope_name)
        tracing_state._traced_module_stack.append(module)

        try:
            result = module.forward(*inputs, **kwargs)
        finally:
            tracing_state.pop_scope()
            tracing_state._traced_module_stack.pop()

        return result

    @staticmethod
    def _get_tracing_name(module, tracing_state):
        if not tracing_state._traced_module_stack:
            return None

        parent_module = tracing_state._traced_module_stack[-1]
        for name, child in parent_module.named_children():
            if child is module:
                return name
        return None


def filter_torch_scope(node):
    """
    Extracts and formats the module name from a PyTorch graph node's scope name.
    """
    scope = node.scopeName().replace('Flatten/', '', 1).replace('Flatten', '', 1)
    scope_list = re.findall(r"\[.*?\]", scope)

    module_name = ''
    if len(scope_list) >= 2:
        module_name = '.'.join(token.strip('[]') for token in scope_list[1:])

    return module_name


def trace(model, args=(), kwargs=None):
    assert kwargs is None, (
        'Keyword arguments are not supported for now. '
        'Please use positional arguments instead!'
    )

    with warnings.catch_warnings(record=True), ScopeNameContextManager():
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

    variables = {}
    for x in graph.nodes():
        for v in list(x.inputs()) + list(x.outputs()):
            if 'tensor' in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=str(v.type()),
                )

    nodes = []
    for x in graph.nodes():
        scope = filter_torch_scope(x)
        node = Node(
            operator=x.kind(),
            attributes={s: getattr(x, x.kindOf(s))(s) for s in x.attributeNames()},
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=scope,
        )
        nodes.append(node)

    graph = Graph(
        name=model.__class__.__module__ + '.' + model.__class__.__name__,
        variables=list(variables.values()),
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )

    return graph
