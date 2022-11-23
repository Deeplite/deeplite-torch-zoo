from torch.hub import load_state_dict_from_url


def load_pretrained_weights(model, checkpoint_url, progress, device):
    pretrained_dict = load_state_dict_from_url(
            checkpoint_url,
            progress=progress,
            check_hash=True,
            map_location=device,
        )
    load_state_dict_partial(model, pretrained_dict)
    return model


def load_state_dict_partial(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()} # pylint: disable=E1135, E1136
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} modules')
