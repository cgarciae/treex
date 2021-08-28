import inspect
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import jax
import jinja2
import yaml

import treex


@dataclass
class Structure:
    obj: tp.Any
    name_path: str
    module_path: str
    members: tp.List[str]


def getinfo():
    module = treex
    name_path = "treex"

    all_members = (
        module.__all__
        if hasattr(module, "__all__")
        else [
            name
            for name, obj in inspect.getmembers(module)
            if (isinstance(obj, ModuleType) and obj.__name__.startswith("treex"))
            or (
                hasattr(obj, "__module__")
                and obj.__class__.__module__ != "typing"
                and "treex" in obj.__module__
                and (inspect.isclass(obj) or inspect.isfunction(obj))
            )
        ]
    )
    all_members = sorted(all_members)

    outputs = {
        name: Structure(
            obj=module,
            name_path=f"{name_path}.{name}",
            module_path=f"{module.__module__}.{name}",
            members=module.__all__ if hasattr(module, "__all__") else [],
        )
        for module, name in ((getattr(module, name), name) for name in all_members)
        if hasattr(module, "__module__")
    }

    return {k: v for k, v in outputs.items() if v}


docs_info = getinfo()

# populate mkdocs
with open("mkdocs.yml", "r") as f:
    docs = yaml.safe_load(f)


[api_reference_index] = [
    index for index, section in enumerate(docs["nav"]) if "API Reference" in section
]


api_reference = jax.tree_map(
    lambda s: s.name_path.replace("treex", "api").replace(".", "/") + ".md", docs_info
)

docs["nav"][api_reference_index] = {"API Reference": api_reference}

with open("mkdocs.yml", "w") as f:
    yaml.safe_dump(docs, f, default_flow_style=False, sort_keys=False)


template = """
# {{name_path}}

::: {{module_path}}
    selection:
        inherited_members: true
        {%- if members %}
        members:
        {%- for member in members %}
            - {{member}}
        {%- endfor %}
        {% endif %}
"""

api_path = Path("docs/api")
shutil.rmtree(api_path, ignore_errors=True)

for structure in jax.tree_leaves(docs_info):
    filepath: Path = api_path / (
        structure.name_path.replace("treex.", "").replace(".", "/") + ".md"
    )
    markdown = jinja2.Template(template).render(
        name_path=structure.name_path,
        module_path=structure.module_path,
        members=structure.members,
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(markdown)
