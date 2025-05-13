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
