#!/usr/bin/env python3

# from functools import lru_cache
import functools
import json
from multiprocessing import Lock
import pickle
import time
import hashlib
import tempfile
from bs4 import BeautifulSoup
from packaging.version import Version
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    BinaryIO,
    Callable,
    Generator,
    List,
    Dict,
    Any,
    Literal,
    Optional,
    Tuple,
)

import instructor
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel, Field

from tqdm import tqdm

import paperext
from paperext.config import CFG


# Define the categorization structure
CATEGORIZATION = {
    "abstract research topics": {
        "3d modeling and reconstruction": {},
        "adversarial machine learning": {},
        "algorithms": {},
        "anomaly detection": {},
        "audio": {},
        "bayesian methods": {},
        "benchmarking": {},
        "calibration and uncertainty estimation": {},
        "causal inference and representation learning": {},
        "classification": {},
        "computer vision": {},
        "data augmentation": {},
        "deep learning and neural networks": {},
        "evaluation": {},
        "game theory": {},
        "generalization bounds": {},
        "generative models": {},
        "genetic algorithms": {},
        "graph-based": {},
        "inference": {},
        "information theory": {},
        "learning paradigms": {},
        "learning theory": {},
        "low-resource machine learning": {},
        "markov chain": {},
        "metaheuristics": {},
        "model calibration": {},
        "modeling": {},
        "natural language processing": {},
        "operation research": {},
        "optimization and algorithms": {},
        "parallelism": {},
        "point cloud processing": {},
        "probabilistic models and inference": {},
        "recommender systems": {},
        "reinforcement learning and decision making": {},
        "statistics": {},
        "time-series": {},
    },
    "application domains": {
        "aerospace and automotive": {},
        "agriculture, climate and environmental science": {},
        "ai ethics and governance": {},
        "application in digital advertising": {},
        "astronomy & astrophysics": {},
        "biomedical and healthcare": {},
        "computational biology and chemistry": {},
        "creativity, intentions and emotions": {},
        "digital innovations": {},
        "engineering and robotics": {},
        "finance and economics": {},
        "human-computer interaction": {},
        "logic": {},
        "mathematics & theory": {},
        "music": {},
        "neuroscience": {},
        "philosophy of science": {},
        "physics and materials science": {},
        "real-time systems": {},
        "sat solving with graph neural networks": {},
        "science and technology studies": {},
        "scientific machine learning": {},
        "social sciences and education": {},
        "telecommunications": {},
        "transportation": {},
    },
    "ignore": {},
}

# List of domains to categorize
DOMAINS_LIST = [
    "drug synergy prediction",
    "question generation",
    "software quality",
    "geriatrics and technology adoption",
    "food web theory",
    "patient care experience",
    "deep learning theory",
    "movie recommendation",
    "credit assignment in reinforcement learning",
    "class-incremental learning",
    "uncertainty",
    "breast cancer treatment",
    "international health",
    "hyperparameter tuning",
    "hcr-ai",
    "post-hoc interpretability",
    "lakatosian philosophy",
    "material science",
    "immunopeptidomics",
    "categorical data",
    "autonomous vehicle simulation",
    "imaging science",
    "low-dose-rate brachytherapy",
    "density estimation",
    "in silico chemistry",
    "neurological sciences",
    "autism research",
    "ageism in ai",
    "structural biology",
    "computational phenotyping",
    "generative flow networks",
    "offline model-based optimization",
    "model safety",
    "low-resource languages",
    "telemedicine",
    "machine learning in genomics",
    "variational inequality problems",
    "conversational question answering",
    "model explainability",
    "implicit likelihood inference",
    "ot",
    "teaching aids",
    "eeg decoding",
    "reconstruction attacks",
    "gaussian processes",
    "neural imaging",
    "decision making",
    "mgwas",
    "surgical education",
    "equivariant neural networks",
]


@dataclass
class CacheSerializer:
    dump: Callable
    load: Callable


@dataclass
class ResponseSerializer:
    def dump(self, response: Response, file_obj: BinaryIO):
        return file_obj.write(response.model_dump_json().encode("utf-8"))

    def load(self, file_obj: BinaryIO) -> Response:
        return Response.model_validate_json(file_obj.read().decode("utf-8"))


@dataclass
class ParsedResponseSerializer(ResponseSerializer):
    content_type: type[ParsedResponse] = ParsedResponse

    def load(self, file_obj: BinaryIO) -> ParsedResponse:
        # filter out the "text" field to avoid the following error:
        # text.format.ResponseFormatText.type
        #   Input should be 'text' [type=literal_error, input_value='json_schema', input_type=str]
        #     For further information visit https://errors.pydantic.dev/2.11/v/literal_error
        # text.format.ResponseFormatTextJSONSchemaConfig.schema
        #   Field required [type=missing, input_value={'name': 'Categorization'...': None, 'strict': True}, input_type=dict]
        #     For further information visit https://errors.pydantic.dev/2.11/v/missing
        # text.format.ResponseFormatJSONObject.type
        #   Input should be 'json_object' [type=literal_error, input_value='json_schema', input_type=str]
        #     For further information visit https://errors.pydantic.dev/2.11/v/literal_error
        data = json.load(file_obj)
        data.pop("text", None)
        return self.content_type.model_validate(data)


@dataclass
class InstructorResponseSerializer(ResponseSerializer):
    content_type: type[BaseModel] = BaseModel

    def dump(self, response: BaseModel, file_obj: BinaryIO):
        model_dump = response.model_dump()
        model_dump["_raw_response"] = response._raw_response.model_dump()
        return file_obj.write(
            json.dumps(model_dump, indent=2, ensure_ascii=False).encode("utf-8")
        )

    def load(self, file_obj: BinaryIO) -> BaseModel:
        data = json.load(file_obj)
        raw_response = data.pop("_raw_response")
        loaded_model = self.content_type.model_validate(data)
        loaded_model._raw_response = ChatCompletion.model_validate(raw_response)
        return loaded_model


@dataclass
class DiskCachedFunc:
    func: Callable
    cache_dir: Path
    serializer: CacheSerializer
    make_key: Callable
    # mem_cache: Callable

    def __init__(
        self,
        func,
        cache_dir: Path = None,
        serializer=pickle,
        make_key=functools.partial(functools._make_key, typed=False),
        # mem_cache: Callable = None,
    ):
        cache_dir = cache_dir or Path(tempfile.gettempdir()) / paperext.__package__
        cache_dir = cache_dir.resolve()
        # mem_cache = mem_cache or lru_cache(func, maxsize=16)

        self._func = func
        self._cache_dir = cache_dir
        self._serializer = serializer
        self._make_key = make_key
        # self._mem_cache = mem_cache
        self._lock = Lock()

    @property
    def info(self):
        return {
            "func": self._func,
            "cache_dir": self._cache_dir,
            "serializer": self._serializer,
            "make_key": self._make_key,
            # "mem_cache": self._mem_cache,
        }

    def exists(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        cache_file = self._cache_dir / f"{self._func.__name__}_{key}"
        return cache_file.exists(), cache_file

    def __call__(self, *args, **kwargs):
        cache_exists, cache_file = self.exists(*args, **kwargs)

        if cache_exists:
            with self._lock:
                return self._serializer.load(cache_file.open("rb"))

        # result = self._mem_cache(*args, **kwargs)
        result = self._func(*args, **kwargs)

        with self._lock:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._serializer.dump(result, cache_file.open("wb"))
        assert self._serializer.load(cache_file.open("rb"))

        return result

    def update(
        self,
        cache_dir: Path = None,
        serializer=None,
        make_key: Callable = None,
        # mem_cache: Callable = None,
    ):
        cache_kwargs = self.info.copy()
        func = cache_kwargs.pop("func")

        if cache_dir is not None:
            cache_kwargs["cache_dir"] = cache_dir
        if serializer is not None:
            cache_kwargs["serializer"] = serializer
        if make_key is not None:
            cache_kwargs["make_key"] = make_key
        # if mem_cache is not None:
        #     cache_kwargs["mem_cache"] = mem_cache

        return DiskCachedFunc(func, **cache_kwargs)


@dataclass
class DiskStoreFunc(DiskCachedFunc):
    element_index: int

    def __init__(self, func: DiskCachedFunc, element_index: int):
        super().__init__(func._func, func._cache_dir, func._serializer, func._make_key)

        self._element_index = element_index

    def exists(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        cache_file = (
            self._cache_dir / f"{self._func.__name__}_{key}_{self._element_index:04d}"
        )
        return cache_file.exists(), cache_file


def disk_cache(
    func,
    cache_dir: Path = None,
    serializer=pickle,
    make_key=functools.partial(functools._make_key, typed=False),
    # mem_cache: Callable = None,
):
    """Cache the result of a function to a file on disk.

    Args:
        func: Function to cache
        cache_dir: Directory to cache the result
        serializer: Serializer to use. Must have a load and dump method
        kwargs: Keyword arguments for lru_cache
    """
    return DiskCachedFunc(func, cache_dir, serializer, make_key)


# Pydantic models for structured output
class DomainCategorization(BaseModel):
    """Pydantic model for domain categorization."""

    selected_domain: str = Field(description="exact domain name from the list")
    parent_category: str = Field(
        description="exact parent category from categorization structure"
    )
    reasoning: str = Field(description="clear reasoning for your categorization")


class Categorization(BaseModel):
    """Pydantic model for categorization."""

    domains_categorization: List[DomainCategorization] = Field(
        description="List of 10 domains categorization"
    )


CATEGORIZATION_VERSIONS = {Version("0.0.0"): Categorization}


@dataclass
class Message:
    type: Literal["system", "user", "assistant"]
    prompt: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        if self.args or self.kwargs:
            return self.prompt.format(*self.args, **self.kwargs)
        return self.prompt

    def format_message(self) -> dict:
        return {
            "role": self.type,
            "content": self.content,
        }


def prompt(
    client: tuple[str, OpenAI],
    messages: list[Message],
    structured_model: tuple[str, BaseModel] = ("", None),
    structured_version: Version = Version("0.0.0"),  # lint: disable=E501
    no_parse: bool = False,
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
    responses_kwargs: dict = {},
) -> Response:
    """Generate a prompt for a list of messages.

    Args:
        client: Tuple of client name and client
        messages: List of messages
        structured_model: Tuple of structured model name and model
        structured_version: Version of the structured model
        no_parse: If True, do not parse the response
        max_attempts: Maximum number of attempts
        check: Function to check if the response is valid
        responses_kwargs: Keyword arguments for the responses.create|parse method
    """
    no_parse = no_parse or not structured_model[1]
    attempt = 0

    response = None
    while attempt < max_attempts and (response is None or not check(response)):
        attempt += 1

        # Generate the response
        if no_parse:
            response = client[1].responses.create(
                input=[m.format_message() for m in messages],
                model=CFG[client[0]].model,
                **responses_kwargs,
            )
        else:
            response = client[1].responses.parse(
                input=[m.format_message() for m in messages],
                model=CFG[client[0]].model,
                text_format=structured_model[1],
                **responses_kwargs,
            )

    # assert check(response), f"Response is not valid: {response}"

    return response


prompt = disk_cache(
    prompt,
    make_key=lambda _args, kwargs: "_".join(
        [
            kwargs["client"][0],
            hashlib.sha256(
                json.dumps(
                    [(m.prompt, m.args, m.kwargs) for m in kwargs["messages"]]
                    + (
                        [kwargs["responses_kwargs"]]
                        if kwargs.get("responses_kwargs", None)
                        else []
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            kwargs["structured_model"][0],
            str(kwargs.get("structured_version", Version("0.0.0"))),
        ]
    ),
)


def instructor_prompt(
    client: tuple[str, instructor.Instructor],
    messages: list[Message],
    structured_model: tuple[str, BaseModel] = ("", None),
    structured_version: Version = Version("0.0.0"),  # lint: disable=E501
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
) -> Response:
    """Generate a prompt for a list of messages.

    Args:
        client: Tuple of client name and client
        messages: List of messages
        structured_model: Tuple of structured model name and model
        structured_version: Version of the structured model
        max_attempts: Maximum number of attempts
        check: Function to check if the response is valid
    """
    attempt = 0

    response = None
    while attempt < max_attempts and (response is None or not check(response)):
        attempt += 1

        # Generate the response
        response, raw_response = client[1].chat.completions.create_with_completion(
            messages=[m.format_message() for m in messages],
            model=CFG[client[0]].model,
            response_model=structured_model[1],
            max_retries=1,
        )

    # assert check(response), f"Response is not valid: {response}"
    response._raw_response = getattr(response, "_raw_response", raw_response)

    return response


instructor_prompt = disk_cache(instructor_prompt, make_key=prompt.info.get("make_key"))


@dataclass
class PerformanceMetrics:
    """Performance metrics for a categorization approach."""

    method_name: str
    success_rate: float
    total_errors: int
    domain_selection_errors: (
        int  # Selected domain from categorization instead of domain list
    )
    category_selection_errors: (
        int  # Selected category from domain list instead of categorization
    )
    parsing_errors: int  # Failed to parse the response
    validation_errors: int  # Response doesn't match expected format
    raw_results: List[Dict[str, Any]] = field(default_factory=list)


class StructuredOutputTester:
    """Test different structured output approaches."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = Path(tempfile.gettempdir()) / "structured_output_comparison"
        self.cache_dir.mkdir(exist_ok=True)

        # Create instructor clients for different modes
        self.instructor_tools_strict_client = instructor.from_openai(
            self.client, mode=instructor.Mode.TOOLS_STRICT
        )
        self.instructor_json_client = instructor.from_openai(
            self.client, mode=instructor.Mode.JSON
        )
        self.instructor_function_client = instructor.from_openai(
            self.client, mode=instructor.Mode.FUNCTIONS
        )

    def _get_cache_key(self, method: str, attempt: int = 0) -> str:
        """Generate cache key for the request."""
        content = f"{method}_{CFG[CFG.platform.select].model}_{attempt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load result from cache if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(result, indent=2))

    def _create_system_prompt(self, format_instructions: str = "") -> str:
        """Create the system prompt for the task."""
        return "\n".join(
            [
                """You are an Expert in Deep Learning Research with extensive knowledge of research domains, methodologies, and their hierarchical relationships. Your task is to accurately categorize Deep Learning Research Domains within a provided hierarchical structure.

### Your Role and Expertise:
- You possess deep understanding of Deep Learning research areas, their interconnections, and hierarchical relationships
- You can identify subtle distinctions between related research domains
- You understand both theoretical foundations and practical applications in Deep Learning
- You are precise in matching domains to their exact categories

### Task Instructions:
1. Analysis Phase:
   - Carefully examine the provided hierarchical structure
   - Review the list of Deep Learning Research Domains to categorize
   - Consider both explicit and implicit relationships between domains
   - Identify any potential ambiguities or edge cases
   - Note any domains that might require special handling (e.g., acronyms, abbreviations)

2. Categorization Process:
   - Select exactly 10 research domains from the provided list that you are MOST CONFIDENT you can accurately categorize
   - For each selected domain, identify the most appropriate parent category from the hierarchical categorization structure
   - Provide clear reasoning for your categorization decisions

3. Quality Requirements:
   - The selected domains MUST be an exact match within the provided list
   - The parent category MUST be an exact match within the hierarchical structure
   - Do not introduce new domains or categories
   - Make selections based on highest confidence
   - Maintain consistency with the existing hierarchical structure
   - A domain that is only composed of a single acronym or abbreviation should be placed with confidence in the "ignore" category
""",
                format_instructions,
            ]
        ).strip()

    def _create_user_prompt(self) -> str:
        """Create the user prompt for the task."""

        # Convert categorization dict to TXT format
        def json_to_txt(d: dict, indent: int = 0) -> str:
            txt = []
            for key, value in sorted(d.items()):
                _open = f"- {key}"
                if isinstance(value, dict):
                    _value = json_to_txt(value, indent + 1)
                else:
                    _value = value

                if not _value:
                    txt.append(f"{'  ' * indent}{_open}")
                else:
                    txt.extend(
                        [f"{'  ' * indent}{entry}" for entry in (f"{_open}:", _value)]
                    )
            return "\n".join([entry for entry in txt if entry.strip()])

        categorization_str = json_to_txt(CATEGORIZATION)
        domains_str = "\n".join([f"- {domain}" for domain in DOMAINS_LIST])

        return f"""Select the top 10 research domains you are most confident about categorizing and provide their categorizations with reasoning.

### The Hierarchical Structure:

{categorization_str}

### List of Deep Learning Research Domains to Categorize:

{domains_str}
"""

    def _validate_result(self, result: Categorization) -> Tuple[int, int, int]:
        """Validate the categorization result and count errors."""
        domain_errors = 0
        category_errors = 0
        validation_errors = 0

        # Check if we have exactly 10 items
        domains = set(
            domain_categorization.selected_domain
            for domain_categorization in result.domains_categorization
        )
        categories = [
            domain_categorization.parent_category
            for domain_categorization in result.domains_categorization
        ]
        if len(domains) != 10 or len(categories) != 10:
            validation_errors += 1

        # Check category selection errors (categories from domain list instead of categorization)
        valid_categories = set(CATEGORIZATION.keys())
        for cat_dict in CATEGORIZATION.values():
            if isinstance(cat_dict, dict):
                valid_categories.update(cat_dict.keys())

        domain_errors += len(domains - set(DOMAINS_LIST))
        category_errors += len(set(categories) - valid_categories)

        return domain_errors, category_errors, validation_errors

    def test_xml_formatting(
        self, num_runs: int = 10
    ) -> Generator[Dict[str, Any], None, None]:
        """Test XML formatting approach."""
        method = "xml_formatting"

        for element_index in range(num_runs):
            cache_key = self._get_cache_key(method, element_index)
            cached_result = self._load_from_cache(cache_key)

            if cached_result:
                cached_result["result"] = (
                    Categorization.model_validate(cached_result["result"])
                    if cached_result["result"] is not None
                    else None
                )
                yield cached_result
                continue

            disk_store_prompt = DiskStoreFunc(
                func=prompt.update(serializer=ResponseSerializer()),
                element_index=element_index,
            )

            system_prompt = self._create_system_prompt(
                """Format your response as XML with the following structure:
<categorization>
  <domains_categorization>
    <domain_categorization>
      <selected_domain>exact domain name from the list</selected_domain>
      <parent_category>exact parent category from categorization structure</parent_category>
      <reasoning>clear reasoning for your categorization</reasoning>
    </domain_categorization>
    <!-- repeat for all 10 domains -->
  </domains_categorization>
</categorization>
"""
            )
            user_prompt = self._create_user_prompt()

            response = None

            try:
                response: Response = disk_store_prompt(
                    client=(CFG.platform.select, self.client),
                    messages=[
                        Message(type="system", prompt=system_prompt),
                        Message(type="user", prompt=user_prompt),
                    ],
                    structured_model=(method, Categorization),
                    no_parse=True,
                    max_attempts=1,
                )

                soup = BeautifulSoup(response.output_text, features="lxml")

                result = Categorization(domains_categorization=[])
                format_errors = 0

                for categorization in soup.find_all("categorization"):
                    for domains_categorization in categorization.find_all(
                        "domains_categorization"
                    ):
                        for domain_categorization in domains_categorization.find_all(
                            "domain_categorization"
                        ):
                            format_errors += 1 - len(
                                domain_categorization.find_all("selected_domain")
                            )
                            format_errors += 1 - len(
                                domain_categorization.find_all("parent_category")
                            )
                            format_errors += 1 - len(
                                domain_categorization.find_all("reasoning")
                            )

                            result.domains_categorization.append(
                                DomainCategorization(
                                    selected_domain=domain_categorization.find(
                                        "selected_domain"
                                    ).text,
                                    parent_category=domain_categorization.find(
                                        "parent_category"
                                    ).text,
                                    reasoning=domain_categorization.find(
                                        "reasoning"
                                    ).text,
                                )
                            )

                domain_errors, category_errors, validation_errors = (
                    self._validate_result(result)
                )

                final_result = {
                    "method": method,
                    "success": True,
                    "result": result,
                    "domain_errors": domain_errors,
                    "category_errors": category_errors,
                    "validation_errors": validation_errors,
                    "format_errors": min(format_errors, 1),
                    "raw_response": response.output_text,
                    "error": None,
                }

            # Add exception handling for pydantic validation error and xml parsing
            except Exception as e:
                final_result = {
                    "method": method,
                    "success": False,
                    "result": None,
                    "domain_errors": 1,
                    "category_errors": 1,
                    "validation_errors": 1,
                    "format_errors": 1,
                    "raw_response": response.output_text if response else None,
                    "error": str(e),
                }

            self._save_to_cache(
                cache_key,
                {
                    **final_result,
                    "result": (
                        final_result["result"].model_dump()
                        if final_result["result"] is not None
                        else None
                    ),
                },
            )
            assert self._load_from_cache(cache_key)

            yield final_result

    def test_json_formatting(
        self,
        num_runs: int = 10,
        method: Literal["json_formatting", "openai_json_object"] = "json_formatting",
    ) -> Generator[Dict[str, Any], None, None]:
        """Test OpenAI with JSON mode (without structured parsing)."""

        match method:
            case "json_formatting":
                responses_kwargs = {}
            case "openai_json_object":
                responses_kwargs = {"text": {"format": {"type": "json_object"}}}

        for element_index in range(num_runs):
            cache_key = self._get_cache_key(method, element_index)
            cached_result = self._load_from_cache(cache_key)

            if cached_result:
                cached_result["result"] = (
                    Categorization.model_validate(cached_result["result"])
                    if cached_result["result"] is not None
                    else None
                )
                yield cached_result
                continue

            disk_store_prompt = DiskStoreFunc(
                func=prompt.update(serializer=ResponseSerializer()),
                element_index=element_index,
            )

            system_prompt = self._create_system_prompt(
                """Respond in JSON format with the following structure:
{
    "domains_categorization": [
        {
            "selected_domain": "domain1", # exact domain name from the list
            "parent_category": "category1", # exact parent category from categorization structure
            "reasoning": "reasoning1" # clear reasoning for your categorization
        },
        ... # repeat for all 10 domains
    ]
}"""
            )
            user_prompt = self._create_user_prompt()

            response = None

            try:
                response: Response = disk_store_prompt(
                    client=(CFG.platform.select, self.client),
                    messages=[
                        Message(type="system", prompt=system_prompt),
                        Message(type="user", prompt=user_prompt),
                    ],
                    structured_model=(method, Categorization),
                    no_parse=True,
                    max_attempts=1,
                    responses_kwargs=responses_kwargs,
                )

                output_lines = response.output_text.strip().splitlines()
                # If output_text startswith '```'
                if output_lines[0].strip().startswith("```"):
                    output_lines = output_lines[1:]
                # If output_text ends with '```'
                if output_lines[-1].strip().endswith("```"):
                    output_lines = output_lines[:-1]

                result = Categorization.model_validate_json("\n".join(output_lines))

                domain_errors, category_errors, validation_errors = (
                    self._validate_result(result)
                )

                final_result = {
                    "method": method,
                    "success": True,
                    "result": result,
                    "domain_errors": domain_errors,
                    "category_errors": category_errors,
                    "validation_errors": validation_errors,
                    "format_errors": 0,
                    "raw_response": response.output_text,
                    "error": None,
                }

            except Exception as e:
                final_result = {
                    "method": method,
                    "success": False,
                    "result": None,
                    "domain_errors": 1,
                    "category_errors": 1,
                    "validation_errors": 1,
                    "format_errors": 1,
                    "raw_response": response.output_text if response else None,
                    "error": str(e),
                }

            self._save_to_cache(
                cache_key,
                {
                    **final_result,
                    "result": (
                        final_result["result"].model_dump()
                        if final_result["result"] is not None
                        else None
                    ),
                },
            )
            assert self._load_from_cache(cache_key)

            yield final_result

    def test_openai_structured(
        self, num_runs: int = 10
    ) -> Generator[Dict[str, Any], None, None]:
        """Test OpenAI structured output with Pydantic."""
        method = "openai_structured"

        for element_index in range(num_runs):
            cache_key = self._get_cache_key(method, element_index)
            cached_result = self._load_from_cache(cache_key)

            if cached_result:
                cached_result["result"] = (
                    Categorization.model_validate(cached_result["result"])
                    if cached_result["result"] is not None
                    else None
                )
                yield cached_result
                continue

            disk_store_prompt = DiskStoreFunc(
                func=prompt.update(
                    serializer=ParsedResponseSerializer(ParsedResponse[Categorization])
                ),
                element_index=element_index,
            )

            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt()

            response = None

            try:
                response: ParsedResponse[Categorization] = disk_store_prompt(
                    client=(CFG.platform.select, self.client),
                    messages=[
                        Message(type="system", prompt=system_prompt),
                        Message(type="user", prompt=user_prompt),
                    ],
                    structured_model=(method, Categorization),
                    no_parse=False,
                    max_attempts=1,
                )

                domain_errors, category_errors, validation_errors = (
                    self._validate_result(response.output_parsed)
                )

                final_result = {
                    "method": method,
                    "success": True,
                    "result": response.output_parsed,
                    "domain_errors": domain_errors,
                    "category_errors": category_errors,
                    "validation_errors": validation_errors,
                    "format_errors": 0,
                    "raw_response": response.output_text,
                    "error": None,
                }

            except Exception as e:
                final_result = {
                    "method": method,
                    "success": False,
                    "result": None,
                    "domain_errors": 1,
                    "category_errors": 1,
                    "validation_errors": 1,
                    "format_errors": 1,
                    "raw_response": response.output_text if response else None,
                    "error": str(e),
                }

            self._save_to_cache(
                cache_key,
                {
                    **final_result,
                    "result": (
                        final_result["result"].model_dump()
                        if final_result["result"] is not None
                        else None
                    ),
                },
            )
            assert self._load_from_cache(cache_key)

            yield final_result

    def test_instructor_json(
        self,
        num_runs: int = 10,
        method: Literal[
            "instructor_json", "instructor_json_strict", "instructor_tools_strict"
        ] = "instructor_json",
    ) -> Generator[Dict[str, Any], None, None]:
        """Test Instructor with JSON mode."""

        match method:
            case "instructor_json":
                client = self.instructor_json_client
            case "instructor_json_strict":
                client = self.instructor_json_schema_client
            case "instructor_tools_strict":
                client = self.instructor_tools_strict_client

        for element_index in range(num_runs):
            cache_key = self._get_cache_key(method, element_index)
            cached_result = self._load_from_cache(cache_key)

            if cached_result:
                cached_result["result"] = (
                    Categorization.model_validate(cached_result["result"])
                    if cached_result["result"] is not None
                    else None
                )
                yield cached_result
                continue

            disk_store_prompt = DiskStoreFunc(
                func=instructor_prompt.update(
                    serializer=InstructorResponseSerializer(content_type=Categorization)
                ),
                element_index=element_index,
            )

            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt()

            response = None

            try:
                response: Response = disk_store_prompt(
                    client=(CFG.platform.select, client),
                    messages=[
                        Message(type="system", prompt=system_prompt),
                        Message(type="user", prompt=user_prompt),
                    ],
                    structured_model=(method, Categorization),
                    max_attempts=1,
                )

                # response = self.instructor_json_client.chat.completions.create(
                #     model=self.model,
                #     response_model=DomainCategorization,
                #     messages=[
                #         {"role": "system", "content": system_prompt},
                #         {"role": "user", "content": user_prompt},
                #     ],
                #     temperature=0.1,
                # )

                domain_errors, category_errors, validation_errors = (
                    self._validate_result(response)
                )

                final_result = {
                    "method": method,
                    "success": True,
                    "result": response,
                    "domain_errors": domain_errors,
                    "category_errors": category_errors,
                    "validation_errors": validation_errors,
                    "format_errors": 0,
                    "raw_response": response._raw_response.choices[0].message.content,
                    "error": None,
                }

            except Exception as e:
                final_result = {
                    "method": method,
                    "success": False,
                    "result": None,
                    "domain_errors": 1,
                    "category_errors": 1,
                    "validation_errors": 1,
                    "format_errors": 1,
                    "raw_response": (
                        response._raw_response.choices[0].message.content
                        if response
                        else None
                    ),
                    "error": str(e),
                }

            self._save_to_cache(
                cache_key,
                {
                    **final_result,
                    "result": (
                        final_result["result"].model_dump()
                        if final_result["result"] is not None
                        else None
                    ),
                },
            )
            assert self._load_from_cache(cache_key)

            yield final_result

    def run_comparison(self, num_runs: int = 10) -> Dict[str, PerformanceMetrics]:
        """Run comparison of all methods."""
        methods = {
            "XML Formatting": self.test_xml_formatting,
            "JSON Formatting": functools.partial(
                self.test_json_formatting, method="json_formatting"
            ),
            "OpenAI JSON Object": functools.partial(
                self.test_json_formatting, method="openai_json_object"
            ),
            "OpenAI Structured": self.test_openai_structured,
            "Instructor JSON": functools.partial(
                self.test_instructor_json, method="instructor_json"
            ),
            # "Instructor JSON Schema": functools.partial(
            #     self.test_instructor_json, method="instructor_json_schema"
            # ),
            "Instructor TOOLS_STRICT": functools.partial(
                self.test_instructor_json, method="instructor_tools_strict"
            ),
            # "Instructor TOOLS_STRICT": self.test_instructor_tools_strict,
            # "Instructor Function Calling": self.test_instructor_function_calling(
            #     num_runs
            # ),
        }

        results = {}

        for method_name, method_func in methods.items():
            method_results = []

            successful_runs = []
            total_domain_errors = 0
            total_category_errors = 0
            total_format_errors = 0
            total_validation_errors = 0

            for result in tqdm(
                method_func(num_runs), total=num_runs, desc=f"Testing {method_name}"
            ):
                method_results.append(result)

                domain_errors = result.get("domain_errors", 0)
                category_errors = result.get("category_errors", 0)
                format_errors = result.get("format_errors", 0)
                validation_errors = result.get("validation_errors", 0)

                total_domain_errors += domain_errors
                total_category_errors += category_errors
                total_format_errors += format_errors
                total_validation_errors += validation_errors

                if (
                    sum(
                        (
                            domain_errors,
                            category_errors,
                            format_errors,
                            validation_errors,
                        )
                    )
                    == 0
                ):
                    successful_runs.append(result)

            success_rate = len(successful_runs) / len(method_results)

            total_errors = (
                total_domain_errors
                + total_category_errors
                + total_format_errors
                + total_validation_errors
            )

            metrics = PerformanceMetrics(
                method_name=method_name,
                success_rate=success_rate,
                total_errors=total_errors,
                domain_selection_errors=total_domain_errors,
                category_selection_errors=total_category_errors,
                parsing_errors=total_format_errors,
                validation_errors=total_validation_errors,
                raw_results=method_results,
            )

            results[method_name] = metrics

        return results

    def print_comparison_report(self, results: Dict[str, PerformanceMetrics]) -> None:
        """Print a detailed comparison report."""
        print("\n" + "=" * 80)
        print("STRUCTURED OUTPUT COMPARISON REPORT")
        print("=" * 80)

        print(
            f"\n{'Method':<25} {'Success Rate':<12} {'Total Errors':<12} {'Parse Errors':<12}"
        )
        print("-" * 80)

        for method_name, metrics in results.items():
            print(
                f"{method_name:<25} {metrics.success_rate:<12.2%} "
                f"{metrics.total_errors:<12} {metrics.parsing_errors:<12}"
            )

        print("\n" + "=" * 80)
        print("DETAILED ERROR BREAKDOWN")
        print("=" * 80)

        for method_name, metrics in results.items():
            print(f"\n{method_name}:")
            print(f"  Success Rate: {metrics.success_rate:.2%}")
            print(f"  Domain Selection Errors: {metrics.domain_selection_errors}")
            print(f"  Category Selection Errors: {metrics.category_selection_errors}")
            print(f"  Parsing Errors: {metrics.parsing_errors}")
            print(f"  Validation Errors: {metrics.validation_errors}")
            print(f"  Total Errors: {metrics.total_errors}")

        # Find best performing method
        best_method = min(
            results.values(),
            key=lambda x: (1 - x.success_rate, x.total_errors),
        )

        print(f"\n{'='*80}")
        print(f"BEST PERFORMING METHOD: {best_method.method_name}")
        print(f"  Success Rate: {best_method.success_rate:.2%}")
        print(f"  Total Errors: {best_method.total_errors}")
        print("=" * 80)


def main():
    """Main function to run the comparison."""
    print("Starting Structured Output Comparison...")

    # Initialize tester
    tester = StructuredOutputTester()

    # Run comparison
    results = tester.run_comparison(num_runs=20)

    # Print report
    tester.print_comparison_report(results)

    # Save results to file
    results_file = Path("structured_output_comparison_results.json")
    with results_file.open("w") as f:
        # Convert results to serializable format
        serializable_results = {}
        for method_name, metrics in results.items():
            serializable_results[method_name] = {
                "method_name": metrics.method_name,
                "success_rate": metrics.success_rate,
                "total_errors": metrics.total_errors,
                "domain_selection_errors": metrics.domain_selection_errors,
                "category_selection_errors": metrics.category_selection_errors,
                "parsing_errors": metrics.parsing_errors,
                "validation_errors": metrics.validation_errors,
                "raw_results": metrics.raw_results,
            }

        json.dump(serializable_results, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
