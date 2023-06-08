import pytest

from deeplite_torch_zoo import get_model_by_name, profile


@pytest.mark.parametrize(
    ('ref_model_name', 'ref_gmacs', 'ref_model_size_mb'),
    [
        ('resnet50', 4.100300288, 102.228128)
    ]
)
def test_profile(
        ref_model_name,
        ref_gmacs,
        ref_model_size_mb,
    ):
    model = get_model_by_name(
        model_name=ref_model_name,
        dataset_name='imagenet',
        pretrained=False,
    )
    model.eval()
    metrics_dict = profile(model)

    assert metrics_dict['GMACs'] == ref_gmacs
    assert metrics_dict['size_Mb'] == ref_model_size_mb
