import pytest

from deeplite_torch_zoo import (get_eval_function, get_model)


@pytest.mark.parametrize(
    ('model_name', 'reference_accuracy'),
    [
        ('mobilenet_v3_small', 0.53125),
        ('squeezenet1_0', 0.46875),
        ('hrnet_w18_small_v2', 0.609375),
        ('efficientnet_es_pruned', 0.609375),
        ('mobilenetv2_w035', 0.375),
        ('mobileone_s0', 0.671875),
    ]
)
def test_classification_model_imagenet_pretrained_accuracy_fast(
        model_name,
        reference_accuracy,
        set_torch_seed_value,
        imagewoof160_dataloaders,
        abs_tolerance=0.05,
    ):
    model = get_model(
        model_name=model_name,
        dataset_name='imagenet',
        pretrained=True,
    )
    eval_fn = get_eval_function(
        model_name=model_name,
        dataset_name='imagenet',
    )
    with set_torch_seed_value():
        top1_accuracy = eval_fn(model, imagewoof160_dataloaders['test'], break_iter=2)['acc']

    assert pytest.approx(top1_accuracy, abs_tolerance) == reference_accuracy
