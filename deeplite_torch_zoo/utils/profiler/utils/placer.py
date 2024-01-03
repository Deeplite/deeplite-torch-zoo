from numpy import prod

from deeplite_torch_zoo.utils.profiler.utils.ir import Layer, Tensor


def get_nodes(graph):
    nodes = []
    for i, node in enumerate(graph.nodes):
        if 'aten' in node.operator:  # aten ops
            inputs = []
            outputs = []
            weights = []
            bias = []

            if 'conv' in node.operator:
                weights = node.inputs[1].shape
                if node.inputs[2].shape is not None:
                    bias = node.inputs[2].shape
                inputs.append(
                    Tensor(
                        name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype,
                        shape=node.inputs[0].shape,
                    )
                )
            elif 'mm' in node.operator:
                weights = node.inputs[2].shape
                if node.inputs[0].shape is not None:
                    bias = node.inputs[0].shape
                inputs.append(
                    Tensor(
                        name=node.inputs[1].name,
                        dtype=node.inputs[1].dtype,
                        shape=node.inputs[1].shape,
                    )
                )
            elif node.operator in ['aten::batch_norm', 'aten::instance_norm']:
                if node.inputs[1].shape is not None:
                    weights = node.inputs[1].shape  # to double-chek
                    bias = node.inputs[2].shape  #
                inputs.append(
                    Tensor(
                        name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype,
                        shape=node.inputs[0].shape,
                    )
                )
            elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
                if node.inputs[2].shape is not None:
                    weights = node.inputs[2].shape  # to double-chek
                    bias = node.inputs[2].shape  # ???
                inputs.append(
                    Tensor(
                        name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype,
                        shape=node.inputs[0].shape,
                    )
                )
            else:
                for x in node.inputs:
                    if x.shape is not None:
                        if x.ndim > 1:
                            inputs.append(
                                Tensor(name=x.name, dtype=x.dtype, shape=x.shape)
                            )

            for x in node.outputs:
                outputs.append(Tensor(name=x.name, dtype=x.dtype, shape=x.shape))

            nodes.append(
                Layer(
                    name="{}_{}".format(i, node.operator),
                    inputs=inputs,
                    outputs=outputs,
                    weights=weights,
                    bias=bias,
                )
            )

    return nodes


class Placer:
    def __init__(self, nodes):
        self.nodes = nodes

    def place(self, num_bytes):
        node_edges = {}
        tensor_shapes = {}

        for node in self.nodes:
            for x in node.inputs:
                if x.name in node_edges:
                    node_edges[x.name] += 1
                else:
                    node_edges[x.name] = 1
                tensor_shapes[x.name] = x.shape

        active_tensors = []
        for node in self.nodes:
            for x in node.outputs:
                if x.name in node_edges:
                    for _ in range(node_edges[x.name]):  # num_edges = lifetime
                        active_tensors.append(x.name)

            malloc = set(active_tensors)  # allocate memory
            ram = sum(prod(tensor_shapes[x]) for x in malloc)
            node.malloc_blocks = malloc
            node.malloc_val = int(ram * num_bytes)

            for x in node.inputs:
                if x.name in malloc:
                    active_tensors.remove(x.name)  # free memory

        return self.nodes
