"""Jinja2-based prompt template loading utilities."""

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@lru_cache(maxsize=1)
def _get_jinja_env() -> Environment:
    """Get cached Jinja2 environment with template directory."""
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=select_autoescape(default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_prompt(template_name: str, block_name: str, **kwargs) -> str:
    """Load and render a Jinja2 prompt template block.

    Args:
        template_name: Template file name (e.g., 'router.j2')
        block_name: Block name within template ('system' or 'user')
        **kwargs: Variables to render in the template

    Returns:
        Rendered prompt string
    """
    env = _get_jinja_env()
    template = env.get_template(template_name)

    # Render the specific block
    block = template.blocks.get(block_name)
    if block is None:
        raise ValueError(f"Block '{block_name}' not found in template '{template_name}'")

    ctx = template.new_context(kwargs)
    return "".join(block(ctx)).strip()
