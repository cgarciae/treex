import inspect
import shutil
import typing as tp
from pathlib import Path
from types import ModuleType

import jax
import jinja2
import yaml

import treex as MODULE

MODULE_NAME = "treex"
INCLUDED_MODULES = {"treex", "treeo"}


class MemberInfo:
    member: tp.Any
    path: tp.Tuple[str]

    def __init__(self, member, path):
        self.member = member
        self.path = path


def getinfo(module: ModuleType, path: tp.Tuple[str, ...]) -> tp.Dict[str, tp.Any]:

    if not hasattr(module, "__all__"):
        return {}

    member_names: tp.List[str]
    member_names = module.__all__
    member_names = sorted(member_names)

    names_paths_values = (
        (name, path + (name,), getattr(module, name)) for name in member_names
    )
    all_members = {
        name: MemberInfo(
            member=getinfo(value, path) if inspect.ismodule(value) else value,
            path=path,
        )
        for name, path, value in names_paths_values
        if value
    }

    return {name: info for name, info in all_members.items() if info.member}


docs_info = getinfo(MODULE, ())


# populate mkdocs
with open("mkdocs.yml", "r") as f:
    docs = yaml.safe_load(f)


[api_reference_index] = [
    index for index, section in enumerate(docs["nav"]) if "API Reference" in section
]


api_reference = jax.tree_map(
    lambda info: "api/" + "/".join(info.path) + ".md",
    docs_info,
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

for info in jax.tree_leaves(docs_info):
    info: MemberInfo

    filepath: Path = api_path / ("/".join(info.path) + ".md")
    markdown = jinja2.Template(template).render(
        name_path=f"{MODULE_NAME}." + ".".join(info.path),
        module_path=f"{MODULE_NAME}." + ".".join(info.path),
        members=[],
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(markdown)
