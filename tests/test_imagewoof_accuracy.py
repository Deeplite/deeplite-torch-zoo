import pytest

from deeplite_torch_zoo import (get_data_splits_by_name, get_eval_function,
                                get_model_by_name)


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
        abs_tolerance=0.05,
        batch_size=32,
    ):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name='imagenet',
        pretrained=True,
        device='cpu',
    )
    eval_fn = get_eval_function(
        model_name=model_name,
        dataset_name='imagenet',
    )
    with set_torch_seed_value():
        dataloaders = get_data_splits_by_name(
            data_root='./',
            model_name=model_name,
            dataset_name='imagewoof_160',
            batch_size=batch_size,
            map_to_imagenet_labels=True
        )
        top1_accuracy = eval_fn(model, dataloaders['test'], break_iter=2)['acc']

    assert pytest.approx(top1_accuracy, abs_tolerance) == reference_accuracy
