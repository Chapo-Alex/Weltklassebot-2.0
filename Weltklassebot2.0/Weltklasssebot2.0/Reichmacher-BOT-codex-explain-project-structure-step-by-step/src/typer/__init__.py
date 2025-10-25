"""Minimal Typer-compatible shim for offline environments."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Union, get_args, get_origin


class BadParameter(Exception):
    """Raised when CLI parsing fails."""


class Exit(Exception):
    """Signal a graceful CLI exit with an exit code."""

    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class OptionInfo:
    default: Any
    help: str = ""
    is_flag: bool = False


def Option(
    default: Any = None,
    *,
    help: str = "",
    **_: Any,
) -> OptionInfo:
    is_flag = isinstance(default, bool)
    return OptionInfo(default=default, help=help, is_flag=is_flag)


class Context:
    """Simplified context compatible with typer.Context."""

    def __init__(self) -> None:
        self.obj: Any = None

    def ensure_object(self, expected_type: type[Any]) -> Any:
        if self.obj is None:
            msg = "Context object has not been initialised"
            raise RuntimeError(msg)
        if not isinstance(self.obj, expected_type):
            msg = f"Context object must be {expected_type.__name__}"
            raise RuntimeError(msg)
        return self.obj


class _Command:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func


class Typer:
    """Very small subset of Typer built on top of ``inspect``."""

    def __init__(self, help: str | None = None, *, add_completion: bool = False) -> None:
        self._help = help
        self._callback: Callable[..., Any] | None = None
        self._commands: Dict[str, _Command] = {}
        self._add_completion = add_completion

    def callback(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._callback = func
            return func

        return decorator

    def command(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cmd_name = name or func.__name__
            self._commands[cmd_name] = _Command(func)
            return func

        return decorator

    def _print_help(self) -> None:
        if self._help:
            print(self._help)
        print("Commands:")
        for name in sorted(self._commands):
            print(f"  {name}")
        if self._callback is not None:
            signature = inspect.signature(self._callback)
            options = [
                f"--{param.name.replace('_', '-')}"
                for param in signature.parameters.values()
                if param.name != "ctx"
            ]
            if options:
                print("Options:")
                for option in options:
                    print(f"  {option}")

    def _parse(
        self,
        func: Callable[..., Any],
        argv: Sequence[str],
        *,
        stop_at_command: bool = False,
    ) -> tuple[dict[str, Any], List[str]]:
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        result: dict[str, Any] = {}
        option_map: dict[str, tuple[str, OptionInfo, inspect.Parameter]] = {}

        for param in params:
            if param.name == "ctx":
                continue
            default = param.default
            if isinstance(default, OptionInfo):
                info = default
            elif default is inspect._empty:
                info = OptionInfo(default=None)
            else:
                info = OptionInfo(default=default)
            option_map[f"--{param.name.replace('_', '-')}"] = (param.name, info, param)
            if info.is_flag:
                result[param.name] = bool(info.default)
            else:
                result[param.name] = info.default

        index = 0
        args = list(argv)
        while index < len(args):
            token = args[index]
            if stop_at_command and token in self._commands:
                break
            if not token.startswith("--"):
                break
            if token not in option_map:
                if stop_at_command and token[2:] in self._commands:
                    break
                msg = f"Unknown option {token}"
                raise BadParameter(msg)
            name, info, param = option_map[token]
            origin = get_origin(param.annotation)
            if info.is_flag:
                result[name] = True
                index += 1
                continue
            if origin in {list, Sequence, Iterable}:
                values: list[str] = []
                index += 1
                while index < len(args) and not args[index].startswith("--"):
                    values.append(args[index])
                    index += 1
                result[name] = values
                continue
            index += 1
            if index >= len(args):
                msg = f"Option {token} requires a value"
                raise BadParameter(msg)
            raw = args[index]
            result[name] = self._convert_value(raw, param.annotation, info.default)
            index += 1

        remaining = args[index:]
        return result, remaining

    def _convert_value(self, raw: str, annotation: Any, default: Any) -> Any:
        origin = get_origin(annotation)
        if origin is Union:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                return self._convert_value(raw, args[0], default)
            return raw
        if annotation in {str, inspect._empty, None}:
            return raw
        if annotation is int:
            return int(raw)
        if annotation is float:
            return float(raw)
        if annotation is bool:
            lowered = raw.lower()
            return lowered in {"1", "true", "yes", "on"}
        if annotation is Path:
            return Path(raw)
        if annotation is datetime:
            value = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        return raw

    def _run(self, argv: Sequence[str]) -> Any:
        ctx = Context()
        remaining = list(argv)
        if "--help" in remaining or "-h" in remaining:
            self._print_help()
            raise Exit(0)
        if self._callback is not None:
            parsed, leftover = self._parse(self._callback, remaining, stop_at_command=True)
            self._callback(ctx, **parsed)
            remaining = leftover
        if not remaining:
            self._print_help()
            raise Exit(0)
        command_name = remaining[0]
        command = self._commands.get(command_name)
        if command is None:
            msg = f"Unknown command {command_name}"
            raise BadParameter(msg)
        cmd_args = remaining[1:]
        parsed, leftover = self._parse(command.func, cmd_args, stop_at_command=False)
        if leftover:
            msg = f"Unexpected arguments: {' '.join(leftover)}"
            raise BadParameter(msg)
        return command.func(ctx, **parsed)

    def __call__(self) -> None:
        try:
            self._run(sys.argv[1:])
        except Exit as exc:
            raise SystemExit(exc.code) from None


__all__ = [
    "BadParameter",
    "Context",
    "Exit",
    "Option",
    "OptionInfo",
    "Typer",
]
