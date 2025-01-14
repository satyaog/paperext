import json
import logging
from pathlib import Path

import pydantic_core
import yaml
from pydantic import BaseModel

from paperext import CFG
from paperext.structured_output.mdl import model_v1, model_v2
from paperext.structured_output.utils import model_dump_yaml
from paperext.utils import split_entry, str_eq


def _model_dump(m):
    if isinstance(m, list):
        return [_model_dump(field) for field in m]

    if isinstance(m, dict):
        return {field_name: _model_dump(field) for field_name, field in m.items()}

    if isinstance(m, BaseModel):
        return m.model_dump()

    return m


def convert_model_v1(extractions: model_v1.PaperExtractions):
    from paperext.structured_output.mdl import model_v2 as dest_model

    fields = {}

    for field_name, field in extractions:
        if field_name in (
            "title",
            "description",
            "type",
        ):
            fields[field_name] = field

        # We want a primary research field as well as a list of
        # sub_research_fields. Each field can have multiple aliases. v1 however
        # has only a single string value for the research_field and a single
        # string value for sub_research_field. Aliases should be extracted from
        # the strings
        elif field_name in ("research_field",):
            name, *aliases = split_entry(
                extractions.research_field.value, sep_left="(", sep_right=")"
            )
            fields["primary_research_field"] = dest_model.ResearchField(
                name={**extractions.research_field.model_dump(), "value": name},
                aliases=aliases,
            )
            logging.info(
                f"Extracted name and aliases from "
                f"{extractions.research_field.value} are: ({name}, {aliases})"
            )

        elif field_name in ("sub_research_field",):
            fields["sub_research_fields"] = []
            sub_research_fields = fields["sub_research_fields"]

            log_msg = (
                f"Extracted name and aliases from "
                f"{extractions.sub_research_field.value} are:"
            )
            for i, srf in enumerate(split_entry(extractions.sub_research_field.value)):
                name, *aliases = split_entry(srf, sep_left="(", sep_right=")")
                srf = dest_model.ResearchField(
                    name={
                        **extractions.sub_research_field.model_dump(),
                        "value": name,
                        **({"justification": "", "quote": ""} if i > 0 else {}),
                    },
                    aliases=aliases,
                )
                sub_research_fields.append(srf)
                log_msg += f" ({name}, {aliases})"

            logging.info(log_msg)

        elif field_name in ("models",):
            fields[field_name] = []
            for m in extractions.models:
                name, *aliases = split_entry(m.name.value, sep_left="(", sep_right=")")
                m = dest_model.Model(
                    name={**m.name.model_dump(), "value": name},
                    aliases=aliases,
                    is_contributed=dest_model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification=f"Role:{[_r.value for _r in model_v1.Role]}",
                        quote=m.role,
                    ).model_dump(),
                    # is_executed is uncertain except when the model is
                    # contributed
                    is_executed=dest_model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification=f"ModelMode:{[_m.value for _m in model_v1.ModelMode]}",
                        quote=m.mode,
                    ).model_dump(),
                    # is_compared is uncertain except when the model is
                    # contributed
                    is_compared=dest_model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification="",
                        quote="",
                    ).model_dump(),
                    referenced_paper_title=dest_model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(m)

                logging.info(
                    f"Extracted name and aliases from "
                    f"{m.name.value} are: ({name}, {aliases})"
                )

        elif field_name in ("datasets",):
            fields[field_name] = []
            for d in extractions.datasets:
                name, *aliases = split_entry(d.name.value, sep_left="(", sep_right=")")
                d = dest_model.Dataset(
                    name={**d.name.model_dump(), "value": name},
                    aliases=aliases,
                    role=d.role,
                    referenced_paper_title=dest_model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(d)

                logging.info(
                    f"Extracted name and aliases from "
                    f"{d.name.value} are: ({name}, {aliases})"
                )

        elif field_name in ("libraries",):
            fields[field_name] = []
            for l in extractions.libraries:
                name, *aliases = split_entry(l.name.value, sep_left="(", sep_right=")")
                l = dest_model.Library(
                    name={**l.name.model_dump(), "value": name},
                    aliases=aliases,
                    role=l.role,
                    referenced_paper_title=dest_model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(l)

                logging.info(
                    f"Extracted name and aliases from "
                    f"{l.name.value} are: ({name}, {aliases})"
                )

    return dest_model.PaperExtractions(**{k: _model_dump(v) for k, v in fields.items()})


def convert_model_v2(extractions: model_v2.PaperExtractions):
    from paperext.structured_output.mdl import model as dest_model

    def convert_enum(enum_cls, value):
        try:
            v = enum_cls(value)
        except ValueError:
            v = enum_cls(str(value).lower().split()[0])
            logging.info(f"Converted enum from {value} to {v}")
        return v

    fields = {}

    for field_name, field in extractions:
        if field_name in (
            "title",
            "description",
            "primary_research_field",
            "sub_research_fields",
        ):
            fields[field_name] = field

        elif field_name in ("type",):
            value = convert_enum(dest_model.ResearchType, extractions.type.value)
            fields[field_name] = {**extractions.type.model_dump(), "value": value}

        elif field_name in ("models",):
            fields[field_name] = []
            for m in extractions.models:
                attributes = [
                    m.is_contributed.value,
                    m.is_executed.value,
                    m.is_compared.value,
                ]
                for i, a in enumerate(attributes):
                    if isinstance(a, bool):
                        continue
                    elif isinstance(a, int):
                        assert a in (0, 1)
                        attributes[i] = a == 1
                    elif isinstance(a, str):
                        assert a.lower().strip() in ("true", "false", "1", "0")
                        attributes[i] = a.lower().strip() == "true"
                    else:
                        assert False

                    logging.info(f"Converted enum from {a} to {attributes[i]}")

                is_contributed, is_executed, is_compared = attributes

                m = dest_model.RefModel(
                    name=m.name.model_dump(),
                    aliases=m.aliases,
                    is_contributed={
                        **m.is_contributed.model_dump(),
                        "value": is_contributed,
                    },
                    is_executed={**m.is_executed.model_dump(), "value": is_executed},
                    is_compared={**m.is_compared.model_dump(), "value": is_compared},
                    referenced_paper_title=m.referenced_paper_title.model_dump(),
                )
                fields[field_name].append(m)

        elif field_name in ("datasets",):
            fields[field_name] = []
            for d in extractions.datasets:
                role = convert_enum(dest_model.Role, d.role)

                d = dest_model.RefDataset(
                    name=d.name.model_dump(),
                    aliases=d.aliases,
                    role=role,
                    referenced_paper_title=d.referenced_paper_title.model_dump(),
                )
                fields[field_name].append(d)

        elif field_name in ("libraries",):
            fields[field_name] = []
            for l in extractions.libraries:
                role = convert_enum(dest_model.Role, l.role)

                l = dest_model.RefDataset(
                    name=l.name.model_dump(),
                    aliases=l.aliases,
                    role=role,
                    referenced_paper_title=l.referenced_paper_title.model_dump(),
                )
                fields[field_name].append(l)

    return dest_model.PaperExtractions(**{k: _model_dump(v) for k, v in fields.items()})


CONVERT_MODEL = {
    model_v1: convert_model_v1,
    model_v2: convert_model_v2,
}


if __name__ == "__main__":
    from paperext.structured_output.mdl import model as dest_model
    from paperext.structured_output.mdl import model_v2 as src_model

    for path in sorted(
        sum(
            map(
                lambda p: sorted(
                    [
                        *p.glob(f"*.json"),
                        *p.glob(f"*.yaml"),
                        *p.glob(f"*/*.json"),
                        *p.glob(f"*/*.yaml"),
                    ]
                ),
                [CFG.dir.merged, CFG.dir.queries],
            ),
            [],
        )
    ):
        path: Path
        model_data = path.read_text()
        try:
            model_data = json.loads(model_data)
        except json.decoder.JSONDecodeError:
            model_data = yaml.safe_load(model_data)

        for model_cls in (
            *[m.PaperExtractions for m in (dest_model, src_model)],
            *[m.ExtractionResponse for m in (dest_model, src_model)],
        ):
            try:
                extractions = model_cls.model_validate(model_data)
                break
            except pydantic_core._pydantic_core.ValidationError as _e:
                e = _e
                logging.warning(
                    f"Failed to validate json {path} of model {model_cls}",
                    exc_info=True,
                )
        else:
            raise e

        try:
            # extractions might be a [dest_model | src_model].ExtractionResponse
            response: src_model.ExtractionResponse = extractions
            extractions = response.extractions
        except AttributeError:
            # extractions is of type [dest_model | src_model].PaperExtractions
            response = None

        if isinstance(extractions, dest_model.PaperExtractions):
            logging.info(f"Model {path.relative_to(CFG.dir.root)} already updated")
            continue

        logging.info(f"Updating {path.relative_to(CFG.dir.root)}")
        extractions = CONVERT_MODEL[src_model](extractions)

        if response is not None:
            src = dest_model.ExtractionResponse(
                paper=response.paper,
                words=response.words,
                extractions=extractions,
                usage=response.usage,
            )
        else:
            src = extractions

        try:
            json.loads(path.read_text())
            model_data = src.model_dump_json(indent=2)
        except json.decoder.JSONDecodeError:
            yaml.safe_load(path.read_text())
            model_data = model_dump_yaml(src)

        path.write_text(model_data)
