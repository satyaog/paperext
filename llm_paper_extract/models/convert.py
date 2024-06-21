from pathlib import Path
import sys

from pydantic import BaseModel
import pydantic_core
from .. import ROOT_DIR
from ..utils import build_validation_set, split_entry, str_eq
from . import model
from . import model_v1


def _model_dump(m):
    if isinstance(m, list):
        return [_model_dump(field) for field in m]

    if isinstance(m, dict):
        return {field_name:_model_dump(field) for field_name, field in m.items()}

    if isinstance(m, BaseModel):
        return m.model_dump()

    return m


def convert_model_v1(extractions:model_v1.PaperExtractions):
    fields = {}

    for field_name, field in extractions:
        if field_name in ("title", "description", "type",):
            fields[field_name] = field
        elif field_name in ("research_field",):
            fields["primary_research_field"] = extractions.research_field
        elif field_name in ("sub_research_field",):
            fields["sub_research_fields"] = [
                model.Explained(
                    value=srf,
                    justification=(
                        extractions.sub_research_field.justification
                        if i == 0
                        else ""
                    ),
                    quote=(
                        extractions.sub_research_field.quote
                        if i == 0
                        else ""
                    ),
                )
                for i, srf in enumerate(split_entry(extractions.sub_research_field.value))
            ]
        elif field_name in ("models",):
            fields[field_name] = [
                model.Model(
                    name=m.name.model_dump(),
                    caracteristics=[m.type.model_dump()],
                    is_executed=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value), justification="", quote=""
                    ).model_dump(),
                    is_compared=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value), justification="", quote=""
                    ).model_dump(),
                    is_contributed=model.Explained(
                        value=str_eq(m.role, model_v1.Role.CONTRIBUTED.value),
                        justification="",
                        quote=""
                    ).model_dump(),
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump()
                )
                for m in extractions.models
            ]
        elif field_name in ("datasets",):
            fields[field_name] = [
                model.Dataset(
                    name=d.name.model_dump(),
                    role=d.role,
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump()
                )
                for d in extractions.datasets
            ]
        elif field_name in ("libraries",):
            fields[field_name] = [
                model.Library(
                    name=l.name.model_dump(),
                    role=l.role,
                    referenced_paper_title=model.Explained(
                        value="", justification="", quote=""
                    ).model_dump()
                )
                for l in extractions.libraries
            ]

    return model.PaperExtractions(
        **{k:_model_dump(v) for k, v in fields.items()}
    )


if __name__ == "__main__":
    validation_set = build_validation_set(ROOT_DIR / "data/")

    annotated = [[], []]
    predictions = [[], []]

    for f in validation_set:
        for path in sorted(
            sum(
                map(
                    lambda p: list(p.glob(f"{f.stem}*.json")),
                    [ROOT_DIR / "data/merged", ROOT_DIR / "data/queries"]
                ),
                []
            )
        ):
            path:Path
            model_json = path.read_text()
            for model_cls in (
                *[m.PaperExtractions for m in (model, model_v1)],
                *[m.ExtractionResponse for m in (model, model_v1)],
            ):
                try:
                    extractions = model_cls.model_validate_json(model_json)
                    break
                except pydantic_core._pydantic_core.ValidationError:
                    pass

            try:
                response:model_v1.ExtractionResponse = extractions
                extractions = response.extractions
            except AttributeError:
                response = None

            if isinstance(extractions, model.PaperExtractions):
                print(f"Model {path.relative_to(ROOT_DIR)} already updated", file=sys.stderr)
                continue

            print(f"Updating {path.relative_to(ROOT_DIR)}", file=sys.stderr)
            extractions = convert_model_v1(extractions)

            if response is not None:
                response = model.ExtractionResponse(
                    paper=response.paper,
                    words=response.words,
                    extractions=extractions,
                    usage=response.usage
                )
                model_json = response.model_dump_json(indent=2)
            else:
                model_json = extractions.model_dump_json(indent=2)

            path.write_text(model_json)
