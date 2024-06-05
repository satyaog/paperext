import random
from pathlib import Path


def build_validation_set(data_dir:Path, seed=42):
    random.seed(seed)

    all_papers = set()
    research_fields = [fn.name.split("_")[0] for fn in data_dir.glob("*_papers.txt")]
    papers_by_field = {}

    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers:set = papers_by_field[field]
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().split("\n")
        all_field_papers = [p for p in all_field_papers if p]
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers)
            all_papers.update(_field_papers)
        print(f"Selected {len(field_papers)} papers out of {len(all_field_papers)} papers for field {field}")

    validation_set = sum(map(list, papers_by_field.values()), [])
    # # Dev validation set
    # validation_set = sum(map(lambda _:random.sample(list(_), 1), papers_by_field.values()), [])
    return list(map(lambda p:data_dir / "cache/arxiv" / p, validation_set))
