import typing
import yaml

from pydantic import BaseModel


def convert_model_json_to_yaml(model_cls: BaseModel, json_data: str, **kwargs):
    model = model_cls.model_validate_json(json_data, **kwargs)
    yaml_data = model_dump_yaml(model)
    assert model_validate_yaml(model_cls, yaml_data) == model
    return yaml_data


def model_dump_yaml(model: BaseModel, **kwargs):
    return yaml.safe_dump(
        model.model_dump(**kwargs, mode="json"),
        width=120,
        allow_unicode=True,
        sort_keys=False,
    )


def model_validate_yaml(model_cls: BaseModel, yaml_data: str, **kwargs):
    return model_cls.model_validate(yaml.safe_load(yaml_data), **kwargs)


def print_model(model_cls: BaseModel, indent=0):
    for field, info in model_cls.model_fields.items():
        print(" " * indent, field, info)
        if typing.get_origin(info.annotation) == list:
            print_model(info.annotation.__args__[0], indent + 2)
            continue

        try:
            print_model(info.annotation, indent + 2)
        except AttributeError:
            pass


def dict_to_txt(d: dict, indent: int = 0) -> str:
    """Convert a dict to TXT format.

    Args:
        d: The dict to convert.
        indent: The indent level.

    Returns:
        The dict in TXT format.
    """
    txt = []
    for key, value in d.items():
        _open = f"- {key}"
        if isinstance(value, dict):
            _value = dict_to_txt(value, indent + 1)
        else:
            _value = value

        if not _value:
            txt.append(f"{'  ' * indent}{_open}")
        else:
            txt.extend([f"{'  ' * indent}{entry}" for entry in (f"{_open}:", _value)])
    return "\n".join([entry for entry in txt if entry.strip()])


def list_to_txt(l: list[str]) -> str:
    """Convert a list to TXT format.

    Args:
        l: The list to convert.

    Returns:
        The list in TXT format.
    """
    return "\n".join([f"- {item}" for item in l])
