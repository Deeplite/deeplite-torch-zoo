import pytest

from deeplite_torch_zoo import get_model, profile


@pytest.mark.parametrize(
    ('ref_model_name', 'ref_gmacs', 'ref_mparams'),
    [
        ('resnet50', 4.100300288, 25.557032)
    ]
)
def test_profile(
        ref_model_name,
        ref_gmacs,
        ref_mparams,
    ):
    model = get_model(
        model_name=ref_model_name,
        dataset_name='imagenet',
        pretrained=False,
    )
    model.eval()
    metrics_dict = profile(model)

    assert metrics_dict['GMACs'] == ref_gmacs
    assert metrics_dict['Mparams'] == ref_mparams
