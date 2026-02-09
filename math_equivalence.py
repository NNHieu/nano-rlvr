import re
from math_verify import parse, verify

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def last_code_output_only_string(string):
    """
    Extract the content of the last <llm-code-output> ... </llm-code-output> block in the string.
    Returns the inner string, or None if no such block exists.
    """
    pattern = r"<llm-code-output>(.*?)</llm-code-output>"
    matches = list(re.finditer(pattern, string, flags=re.DOTALL))
    if matches:
        # Return the inner content (stripped)
        return matches[-1].group(1).strip()
    return None

def extract_boxed_answer(output, mode='gen'):
    if "</think>" in output:
        output = output.split("</think>")[1]
    string_in_last_boxed = last_boxed_only_string(output)
    return remove_boxed(string_in_last_boxed) if string_in_last_boxed is not None else ''

def extract_code_answer(output, mode='gen'):
    if "</think>" in output:
        output = output.split("</think>")[1]
    string_in_last_codeoutput = last_code_output_only_string(output)
    return string_in_last_codeoutput if string_in_last_codeoutput is not None else ''

def is_math_equiv(gold, answer, parsing_timeout=1.0):
    gold = parse(gold)
    answer = parse(answer, parsing_timeout=parsing_timeout)
    return verify(gold, answer)