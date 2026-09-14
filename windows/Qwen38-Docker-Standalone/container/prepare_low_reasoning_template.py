#!/usr/bin/env python3
"""Create a non-destructive chat template with low reasoning as the default."""

from pathlib import Path


SOURCE = Path("/model/chat_template.jinja")
TARGET = Path("/tmp/qwen38_chat_template_low_reasoning.jinja")
ANCHOR = "reasoning_effort|default('xhigh')"
REPLACEMENT = "reasoning_effort|default('low')"


def main() -> None:
    template = SOURCE.read_text(encoding="utf-8")
    matches = template.count(ANCHOR)
    if matches != 1:
        raise RuntimeError(
            f"Refusing to prepare {TARGET}: expected one reasoning-default "
            f"anchor in {SOURCE}, found {matches}."
        )
    TARGET.write_text(template.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    print("Thinking mode: enabled; default reasoning effort: low")
    print(f"Using container-local chat template: {TARGET}")


if __name__ == "__main__":
    main()
