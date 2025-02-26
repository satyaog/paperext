import argparse

import pandas as pd
from paperext.config import CFG
from paperext.structured_output import get_struct_module


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cost_input",
        type=float,
        metavar="FLOAT",
        help=f"Cost per {1e6} input tokens",
    )
    parser.add_argument(
        "cost_output",
        type=float,
        metavar="FLOAT",
        help=f"Cost per {1e6} input tokens",
    )
    options = parser.parse_args(argv)

    cost_input = options.cost_input / 1e6
    cost_output = options.cost_output / 1e6

    in_tokens = []
    out_tokens = []

    for response in (
        CFG.dir.data / CFG.platform.struct / "queries" / CFG.platform.select
    ).glob("*.json"):
        response = get_struct_module(
            CFG.platform.struct
        ).model.Response.model_validate_json(response.read_text())
        in_tokens.append(response.usage["prompt_tokens"])
        out_tokens.append(response.usage["completion_tokens"])

    sum_input = sum(in_tokens) * cost_input
    sum_output = sum(out_tokens) * cost_output

    data = {
        f"Total ({len(in_tokens)})": [sum_input + sum_output],
        f"Average": [(sum_input + sum_output) / len(in_tokens)],
        f"{sum(in_tokens)} input token(s) @{options.cost_input:.2f}$/1M": [sum_input],
        f"{sum(out_tokens)} output token(s) @{options.cost_output:.2f}$/1M": [
            sum_output
        ],
    }
    df = pd.DataFrame(data).round(3)

    string = "\n".join(
        f"{l} $" for l in df.transpose().to_string(header=False).splitlines()
    )
    print(string)


# Total (284)   2.047153
# avg           0.007208
# 149813 input  0.374533
# 167262 output 1.67262


# print(f"For {len(in_tokens)} queries:")
# print("Total: ", f"{sum_input + sum_output:.3f}", sep="\t")
# print("avg:   ", f"{(sum_input + sum_output) / len(in_tokens):.3f}", sep="\t")
# print(f"{sum(in_tokens)}\tinput: ", f"{sum_input:.3f}", sep="\t")
# print(f"{sum(out_tokens)}\toutput:", f"{sum_output:.3f}", sep="\t")


if __name__ == "__main__":
    main()
