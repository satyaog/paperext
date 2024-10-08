#################
llm_paper_extract
#################

**********************************
llm_paper_extract.download_convert
**********************************

``pdftotext`` from `https://poppler.freedesktop.org/`_ is required to run this utility.

::

    usage: python3 -m llm_paper_extract.download_convert [-h] [--cache-dir DIR] JSON

    Utility to download and convert a list of papers' pdfs -> "txts.

    stdout will contain the list of files successfully converted separated by '\n'.

    positional arguments:
      JSON             Paperoni json output of papers to download and convert pdfs -> txts

    options:
      -h, --help       show this help message and exit
      --cache-dir DIR  Directory to store downloaded and converted pdfs -> txts

    Example:
      $ python3 -m llm_paper_extract.download_convert paperoni-2024-07-04.json
        [DEBUG]
        data/cache/arxiv/1901.07186.txt
        data/cache/arxiv/1906.05433.txt
        ...
        data/cache/html/874f823e6462acbbb07cc57d32e09217.txt
        data/cache/html/8a46fcbc0c34ea85102920cba7039290.txt
        ...
        data/cache/openreview/0k_DN90uWF.txt
        data/cache/openreview/2Q8TZWAHv4.txt
        ...
        data/cache/pdf/80c62591b54231aa42e4418fe3d45e8f.txt
        data/cache/pdf/81709b4783324a59fd2632ee694e9071.txt
        ...
        Successfully downloaded and converted 587 out of 867 papers
        arxiv:455/455
        html:3/3
        openreview:52/52
        pdf:77/145
      $ python3 -m llm_paper_extract.download_convert paperoni-2024-07-04.json > data/query_set.txt
        [DEBUG]
        Successfully downloaded and converted 587 out of 867 papers
        arxiv:455/455
        html:3/3
        openreview:52/52
        pdf:77/145

***********************
llm_paper_extract.query
***********************

An OpenAI API Key is required to run this utility:

::

    export OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

::

    usage: python3 -m llm_paper_extract.query [-h] [--papers [PAPERS ...]] [--input TXT]

    Utility to query Chat-GPT on papers

    options:
      -h, --help            show this help message and exit
      --papers [PAPERS ...]
                            Papers to analyse
      --input TXT           List of papers to analyse

    Example:
      $ python3 -m llm_paper_extract.query --input data/query_set.txt > query.out 2> query.err

*****************************************
llm_paper_extract.parse_validation_errors
*****************************************

::

    usage: python3 -m llm_paper_extract.parse_validation_errors [-h] TXT

    Utility to parse and gather stats on query logs

    positional arguments:
      TXT         query stdout file to parse

    options:
      -h, --help  show this help message and exit

    Example:
      $ python3 -m llm_paper_extract.parse_validation_errors query.err
        Per paper errors
        ================
        Failures count for paper: 1
        ---------------------------
        primary_research_field.aliases      1 ~1
        primary_research_field.name         1 ~1
        sub_research_fields.*.aliases       2 ~1
        sub_research_fields.*.name          2 ~1

        [...]
        Failures count for paper: 2
        ---------------------------
        datasets                            1 ~1
        models.*                           13 ~1
        sub_research_fields.*.aliases       1 ~1
        sub_research_fields.*.name          1 ~1

        libraries.*.aliases                 1 ~1
        libraries.*.referenced_paper_title  1 ~1
        libraries.*.role                    1 ~1

        [...]
        Failures count for paper: 2
        ---------------------------
        Generic Error: 
          Invalid JSON: control character (\u0000-\u001F) found while parsing a string at line 1 column 2885 [...]
            For further information visit https://errors.pydantic.dev/2.7/v/json_invalid,)

        Generic Error: 
          Invalid JSON: control character (\u0000-\u001F) found while parsing a string at line 1 column 2885 [...]
            For further information visit https://errors.pydantic.dev/2.7/v/json_invalid,)

        [...]
        Error type involved in a paper validation
        =========================================
        Generic Error                       2 /62
        datasets                            5 /62
        libraries.*.aliases                10 /62
        libraries.*.referenced_paper_title  9 /62
        libraries.*.role                    9 /62
        models.*                            3 /62
        primary_research_field.aliases     40 /62
        primary_research_field.name        42 /62
        sub_research_fields.*.aliases      32 /62
        sub_research_fields.*.name         37 /62

        Queries Stats
        =============
        Requests cnt                          204
        Errors cnt                            62
        1 failure(s)                          46
        2 failure(s)                          8

******************************
llm_paper_extract.merge_papers
******************************

**************************
llm_paper_extract.analysis
**************************

::

    usage: python3 -m llm_paper_extract.analysis [-h] [--papers [PAPERS ...]] [--input TXT]

    Utility to analyses Chat-GPT responses on papers

    Confidence and multi-label confidence matrices will be dumped into data/analysis

    options:
      -h, --help            show this help message and exit
      --papers [PAPERS ...]
                            Papers to analyse
      --input TXT           List of papers to analyse

    Example:
      $ python3 -m llm_paper_extract.analysis
