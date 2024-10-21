import logging
import sys
from pathlib import Path

import pydantic_core
from pydantic import BaseModel

from paperext import ROOT_DIR
from paperext.models import model, model_v1
from paperext.utils import build_validation_set, split_entry, str_eq


def _model_dump(m):
    if isinstance(m, list):
        return [_model_dump(field) for field in m]

    if isinstance(m, dict):
        return {field_name: _model_dump(field) for field_name, field in m.items()}

    if isinstance(m, BaseModel):
        return m.model_dump()

    return m


def convert_model_v1(extractions: model_v1.PaperExtractions):
    fields = {}

    for field_name, field in extractions:
        if field_name in (
            "title",
            "description",
            "type",
        ):
            fields[field_name] = field

        elif field_name in ("research_field",):
            name, *aliases = split_entry(
                extractions.research_field.value, sep_left="(", sep_right=")"
            )
            fields["primary_research_field"] = model.ResearchField(
                name={**extractions.research_field.model_dump(), "value": name},
                aliases=aliases,
            )

        elif field_name in ("sub_research_field",):
            fields["sub_research_fields"] = []
            sub_research_fields = fields["sub_research_fields"]
            for i, srf in enumerate(split_entry(extractions.sub_research_field.value)):
                name, *aliases = split_entry(srf, sep_left="(", sep_right=")")
                srf = model.ResearchField(
                    name={
                        **extractions.sub_research_field.model_dump(),
                        "value": name,
                        **({"justification": "", "quote": ""} if i > 0 else {}),
                    },
                    aliases=aliases,
                )
                sub_research_fields.append(srf)

        elif field_name in ("models",):
            fields[field_name] = []
            for m in extractions.models:
                name, *aliases = split_entry(m.name.value, sep_left="(", sep_right=")")
                m = model.Model(
                    name={**m.name.model_dump(), "value": name},
                    aliases=aliases,
                    is_contributed=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification=f"Role:{[_r.value for _r in model_v1.Role]}",
                        quote=m.role,
                    ).model_dump(),
                    # is_executed is uncertain except when the model is
                    # contributed
                    is_executed=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification=f"ModelMode:{[_m.value for _m in model_v1.ModelMode]}",
                        quote=m.mode,
                    ).model_dump(),
                    # is_compared is uncertain except when the model is
                    # contributed
                    is_compared=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification="",
                        quote="",
                    ).model_dump(),
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(m)

        elif field_name in ("datasets",):
            fields[field_name] = []
            for d in extractions.datasets:
                name, *aliases = split_entry(d.name.value, sep_left="(", sep_right=")")
                d = model.Dataset(
                    name={**d.name.model_dump(), "value": name},
                    aliases=aliases,
                    role=d.role,
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(d)

        elif field_name in ("libraries",):
            fields[field_name] = []
            for l in extractions.libraries:
                name, *aliases = split_entry(l.name.value, sep_left="(", sep_right=")")
                l = model.Library(
                    name={**l.name.model_dump(), "value": name},
                    aliases=aliases,
                    role=l.role,
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump(),
                )
                fields[field_name].append(l)

    return model.PaperExtractions(**{k: _model_dump(v) for k, v in fields.items()})


if __name__ == "__main__":
    validation_set = build_validation_set(ROOT_DIR / "data/")

    annotated = [[], []]
    predictions = [[], []]

    for f in sorted(validation_set):
        for path in sorted(
            sum(
                map(
                    lambda p: sorted(p.glob(f"{f.stem}*.json")),
                    [ROOT_DIR / "data/merged", ROOT_DIR / "data/queries"],
                ),
                [],
            )
        ):
            path: Path
            model_json = path.read_text()
            for model_cls in (
                *[m.PaperExtractions for m in (model, model_v1)],
                *[m.ExtractionResponse for m in (model, model_v1)],
            ):
                try:
                    extractions = model_cls.model_validate_json(model_json)
                    break
                except pydantic_core._pydantic_core.ValidationError as _e:
                    e = _e
                    logging.warning(
                        f"Failed to validate json of model {model_cls}: {e}",
                        exc_info=True,
                    )
            else:
                raise e

            try:
                # extractions might be a [model | model_v1].ExtractionResponse
                response: model_v1.ExtractionResponse = extractions
                extractions = response.extractions
            except AttributeError:
                # extractions is of type [model | model_v1].PaperExtractions
                response = None

            if isinstance(extractions, model.PaperExtractions):
                logging.info(f"Model {path.relative_to(ROOT_DIR)} already updated")
                continue

            logging.info(f"Updating {path.relative_to(ROOT_DIR)}")
            extractions = convert_model_v1(extractions)

            if response is not None:
                response = model.ExtractionResponse(
                    paper=response.paper,
                    words=response.words,
                    extractions=extractions,
                    usage=response.usage,
                )
                model_json = response.model_dump_json(indent=2)
            else:
                model_json = extractions.model_dump_json(indent=2)

            path.write_text(model_json)
