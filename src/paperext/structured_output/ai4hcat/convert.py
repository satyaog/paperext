import json
import logging
from pathlib import Path

import pydantic_core
import yaml
from pydantic import BaseModel

from paperext import CFG
from paperext.structured_output.ai4hcat import model_v1, model_v2, model
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
    from paperext.structured_output.ai4hcat import model_v2 as dest_model

    return dest_model.PaperExtractions.model_validate_json(
        extractions.model_dump_json()
    )


def convert_model_v2(extractions: model_v2.PaperExtractions):
    from paperext.structured_output.ai4hcat import model as dest_model

    fields = {}

    for field_name, field in extractions:
        fields[field_name] = field

    fields["sustainable_development_is_central"] = dest_model.Explained[bool](
        value=False, justification="", quote=""
    )

    return dest_model.PaperExtractions(**{k: _model_dump(v) for k, v in fields.items()})


CONVERT_MODEL = {
    model_v1: convert_model_v1,
    model_v2: convert_model_v2,
}


if __name__ == "__main__":
    from paperext.structured_output.ai4hcat import model as dest_model
    from paperext.structured_output.ai4hcat import model_v2 as src_model

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
            *[m.Response for m in (dest_model, src_model)],
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
            # extractions might be a [dest_model | src_model].Response
            response: src_model.Response = extractions
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
            src = dest_model.Response(
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
