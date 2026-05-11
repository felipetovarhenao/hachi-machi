
from hachi_machi.cli import main
from hachi_machi.operations import DataAugmenter
import itertools
import os
import re
import textwrap
from pathlib import Path
import click
import inspect

CLI_CMD = 'hxmx'


class AutoDoc:
    def __init__(self,
                 cli: click.Command,
                 output_dir: str | os.PathLike,
                 cli_name: str,
                 ) -> None:
        self.cli = cli
        self.output_dir = Path(output_dir)
        self.cli_name = cli_name or cli.name or "cli"

    def generate(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        counter = itertools.count(1)
        self._walk(self.cli, self.cli_name, [], self.output_dir, counter)
        self.ops_docs()

        print(f"Docs written to: {self.output_dir.resolve()}")

    def _walk(self,
              cmd: click.Command,
              name: str,
              parents: list[str],
              output_dir: Path,
              counter: itertools.count,) -> None:

        pos = next(counter) + 1 if len(parents) > 0 else 1

        if isinstance(cmd, click.Group):
            subdir = output_dir / name if parents else output_dir
            subdir.mkdir(parents=True, exist_ok=True)

            subcommand_names = list(getattr(cmd, "commands", {}).keys())
            index_md = self._render_group_index(group=cmd,
                                                name=name,
                                                parents=parents,
                                                sidebar_position=pos,
                                                subcommand_names=subcommand_names)
            (subdir / "index.md").write_text(index_md, encoding="utf-8")

            new_parents = parents + [name] if parents else [name]
            child_counter = itertools.count(1)
            for sub_name, sub_cmd in (getattr(cmd, "commands", {}) or {}).items():
                self._walk(sub_cmd, sub_name, new_parents,
                           subdir, child_counter)
        else:
            md = self._render_command(cmd, name,  pos)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{name}.md").write_text(md, encoding="utf-8")

    @staticmethod
    def _render_command(cmd: click.Command,
                        name: str,
                        sidebar_position: int,) -> str:
        full_name = name
        description = textwrap.dedent(cmd.help or "").strip()
        short_desc = description.split("\n")[0] if description else ""

        sections = [
            AutoDoc._frontmatter(
                full_name, sidebar_position, short_desc),
            "",
            f"# `{full_name}`",
            "",
        ]

        if description:
            sections += [description, ""]

        ctx = click.Context(cmd, info_name=full_name)
        usage = " ".join([CLI_CMD] + ctx.command_path.split()) + \
            AutoDoc._usage_suffix(cmd)
        sections += ["## Usage", "", AutoDoc._code_block(usage), ""]

        arguments = [p for p in cmd.params if isinstance(p, click.Argument)]
        options = [p for p in cmd.params if isinstance(p, click.Option)]

        if arguments:
            sections += [
                "## Arguments", "",
                "| Name | Type | Required | Default | Description |",
                "|------|------|:--------:|---------|-------------|",
                *AutoDoc._param_rows(arguments),
                "",
            ]

        if options:
            sections += [
                "## Options", "",
                "| Option | Type | Required | Default | Description |",
                "|--------|------|:--------:|---------|-------------|",
                *AutoDoc._param_rows(options),
                "",
            ]

        epilog = textwrap.dedent(cmd.epilog or "").strip()
        if epilog:
            sections += ["## Notes", "", epilog, ""]

        return "\n".join(sections)

    @staticmethod
    def _render_group_index(group: click.Group,
                            name: str,
                            parents: list[str],
                            sidebar_position: int,
                            subcommand_names: list[str],) -> str:
        full_name = " ".join(parents + [name]) if parents or name else name
        description = textwrap.dedent(group.help or "").strip()
        short_desc = description.split("\n")[0] if description else ""

        sections = [
            AutoDoc._frontmatter(
                full_name or "CLI Reference", sidebar_position, short_desc),
            "",
            f"# `{full_name}`" if full_name else "# CLI Reference",
            "",
        ]

        if description:
            sections += [description, ""]

        if subcommand_names:
            sections += ["## Commands", ""]
            for sub in sorted(subcommand_names):
                sub_cmd = group.commands.get(sub)
                sub_help = (sub_cmd.help.strip() or "").split(
                    "\n")[0] if sub_cmd else ""
                entry = f"- [`{sub}`](./{sub}) — {sub_help}" if sub_help else f"- [`{sub}`](./{sub})"
                sections.append(entry)
            sections.append("")

        return "\n".join(sections)

    @staticmethod
    def _frontmatter(title: str,
                     sidebar_position: int,
                     description: str = "") -> str:
        lines = ["---", f"title: {title}",
                 f"sidebar_position: {sidebar_position}"]
        if description:
            clean = re.sub(r"[`*_]", "", description.split("\n")[0])
            lines.append(f'description: "{clean}"')
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _code_block(content: str, lang: str = "bash") -> str:
        return f"```{lang}\n{content}\n```"

    @staticmethod
    def _param_rows(params: list) -> list[str]:
        rows = []
        for p in params:
            if isinstance(p, click.Option):
                name = ", ".join(p.opts)
                required = "✓" if p.required else ""
                default = f"`{p.default}`" if p.default is not None else "—"
                multiple = " *(multiple)*" if p.multiple else ""
                type_label = AutoDoc._type_name(p.type) + multiple
                help_text = (p.help or "").replace("|", "\\|")
                rows.append(
                    f"| `{name}` | {type_label} | {required} | {default} | {help_text} |")
            elif isinstance(p, click.Argument):
                required = "✓" if p.required else ""
                multiple = " *(multiple)*" if p.nargs == -1 else ""
                type_label = AutoDoc._type_name(p.type) + multiple
                rows.append(
                    f"| `{p.human_readable_name}` | {type_label} | {required} | — | *(positional argument)* |")
        return rows

    @staticmethod
    def _usage_suffix(cmd: click.Command) -> str:
        parts = []
        for p in cmd.params:
            if isinstance(p, click.Argument):
                if p.nargs == -1:
                    parts.append(f"[{p.human_readable_name}]...")
                elif p.required:
                    parts.append(f"<{p.human_readable_name}>")
                else:
                    parts.append(f"[{p.human_readable_name}]")
            elif isinstance(p, click.Option):
                flag = p.opts[-1]
                if p.is_flag:
                    parts.append(f"[{flag}]")
                elif p.required:
                    parts.append(f"{flag} <{p.type.name.upper()}>")
                else:
                    parts.append(f"[{flag} <{p.type.name.upper()}>]")
        suffix = " ".join(parts)
        return f" {suffix}" if suffix else ""

    @staticmethod
    def _type_name(param_type: click.ParamType) -> str:
        type_map = {
            click.STRING: "string",
            click.INT: "integer",
            click.FLOAT: "float",
            click.BOOL: "boolean",
            click.UUID: "UUID",
        }
        for known, label in type_map.items():
            if param_type is known:
                return label

        if isinstance(param_type, click.Choice):
            return "choice (" + "\\|".join(f"`{c}`" for c in param_type.choices) + ")"
        if isinstance(param_type, click.Path):
            parts = []
            if param_type.exists:
                parts.append("must exist")
            if param_type.file_okay and not param_type.dir_okay:
                parts.append("file")
            elif param_type.dir_okay and not param_type.file_okay:
                parts.append("directory")
            suffix = f" — {', '.join(parts)}" if parts else ""
            return f"path{suffix}"
        if isinstance(param_type, click.File):
            return f"file (mode: `{param_type.mode}`)"
        if isinstance(param_type, click.IntRange):
            lo = param_type.min if param_type.min is not None else "−∞"
            hi = param_type.max if param_type.max is not None else "+∞"
            return f"integer [{lo}, {hi}]"
        if isinstance(param_type, click.FloatRange):
            lo = param_type.min if param_type.min is not None else "−∞"
            hi = param_type.max if param_type.max is not None else "+∞"
            return f"float [{lo}, {hi}]"
        if isinstance(param_type, click.Tuple):
            return "tuple (" + ", ".join(AutoDoc._type_name(t) for t in param_type.types) + ")"

        return type(param_type).__name__.lower()

    def ops_docs(self) -> str:
        cls = DataAugmenter
        BASE_ARGS = {
            "dims": (
                "Feature indices to apply the operation to. "
                "Use `t` for the time dimension (temporal datasets only). "
                "If omitted, the operation is applied to all features."
            ),
            "p": "Probability of applying the operation. Default: `1.0`.",
        }

        def parse_args_section(docstring: str) -> dict[str, str]:
            if not docstring:
                return {}
            match = re.search(r"Args:\s*\n((?:[ \t]+.+\n?)+)", docstring)
            if not match:
                return {}
            args = {}
            current_key = None
            for line in match.group(1).splitlines():
                kv = re.match(r"^\s*(\w+):\s*(.+)", line)
                if kv:
                    current_key = kv.group(1)
                    args[current_key] = kv.group(2).strip()
                elif current_key and re.match(r"^\s{8,}", line):
                    args[current_key] += " " + line.strip()
            return args

        def class_description(cls) -> str:
            doc = inspect.getdoc(cls)
            if not doc:
                return ""
            return doc.strip()
        op_docs = {}

        for op_name, op_cls in cls.OPERATIONS.items():
            lines = []
            signature_params = cls.get_signature(op_cls)
            sig_parts = []
            for name, param in signature_params.items():
                if name == 'dims':
                    name = f"*{name}"
                if param.default is inspect.Parameter.empty:
                    sig_parts.append(name)
                else:
                    default = param.default
                    sig_parts.append(f"{name}={default}")
            signature = f"\n\n```rust\n{op_name}({', '.join(sig_parts)})\n```\n"

            lines.append(f"{signature}\n")

            desc = class_description(op_cls)
            if desc:
                lines.append(f"{desc}\n")

            class_args = op_cls.docs()

            if class_args:
                lines.append("### Arguments\n")
                for arg_name, arg_desc in class_args.items():
                    lines.append(f"- `{arg_name}`: {arg_desc}")
                lines.append("")
            op_docs[op_name] = lines

        ops_dir = self.output_dir / 'operations/'
        ops_dir.mkdir(exist_ok=True)

        for name, lines in op_docs.items():
            (ops_dir / f'{name}.md').write_text("\n".join(lines))


if __name__ == "__main__":
    AutoDoc(cli=main,
            output_dir='./docs/commands/',
            cli_name="Documentation").generate()
