"""
Inquira TUI — Rich-powered terminal user interface.

All display logic lives here. The SmartQABot class in main.py stays
purely functional; this module wraps its output in styled Rich panels,
tables, spinners, and prompts.
"""

import datetime
import json
import os
import readline
import select
import sys
import termios
import tty

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────────────

INQUIRA_THEME = Theme(
    {
        "header": "bold bright_cyan",
        "subheader": "bold bright_magenta",
        "success": "bold green",
        "error": "bold red",
        "warning": "bold yellow",
        "info": "bold bright_blue",
        "muted": "dim white",
        "accent": "bold bright_yellow",
        "prompt": "bold cyan",
        "confidence.high": "bold green",
        "confidence.medium": "bold yellow",
        "confidence.low": "bold red",
        "key": "bold bright_white",
        "value": "bright_white",
    }
)

console = Console(theme=INQUIRA_THEME)

# ── ASCII Art Logo & Chevron ─────────────────────────────────────────────────
# 8 distinct alphabets, each with its own dedicated color transitioning from lighter to darker.

LOGO_GLYPHS = [
    # 0: Chevron >
    [
        r"██╗  ",
        r"╚██╗ ",
        r" ╚██╗",
        r" ██╔╝",
        r"██╔╝ ",
        r"╚═╝  ",
    ],
    # 1: I
    [
        r"██╗",
        r"██║",
        r"██║",
        r"██║",
        r"██║",
        r"╚═╝",
    ],
    # 2: N
    [
        r"███╗   ██╗",
        r"████╗  ██║",
        r"██╔██╗ ██║",
        r"██║╚██╗██║",
        r"██║ ╚████║",
        r"╚═╝  ╚═══╝",
    ],
    # 3: Q
    [
        r" ██████╗ ",
        r"██╔═══██╗",
        r"██║   ██║",
        r"██║▄▄ ██║",
        r"╚██████╔╝",
        r" ╚══▀▀═╝ ",
    ],
    # 4: U
    [
        r"██╗   ██╗",
        r"██║   ██║",
        r"██║   ██║",
        r"██║   ██║",
        r"╚██████╔╝",
        r" ╚═════╝ ",
    ],
    # 5: I
    [
        r"██╗",
        r"██║",
        r"██║",
        r"██║",
        r"██║",
        r"╚═╝",
    ],
    # 6: R
    [
        r"██████╗ ",
        r"██╔══██╗",
        r"██████╔╝",
        r"██╔══██╗",
        r"██║  ██║",
        r"╚═╝  ╚═╝",
    ],
    # 7: A
    [
        r" █████╗ ",
        r"██╔══██╗",
        r"███████║",
        r"██╔══██║",
        r"██║  ██║",
        r"╚═╝  ╚═╝",
    ],
]

# ── Color Palette Definition ────────────────────────────────────────────────
# Smooth pastel lavender-to-orchid color scheme for each alphabet glyph.

logo_colors = [
    "#B4ABFF",  # >
    "#BCA7FD",  # I
    "#C3A2F8",  # N
    "#CB9CF2",  # Q
    "#D194EB",  # U
    "#D78BE3",  # I
    "#D983DA",  # R
    "#D17ACF",  # A
]

LOGO_COLORS = logo_colors
ALPHABET_COLORS = logo_colors
GRADIENT_STOPS = [(i / max(1, len(logo_colors) - 1), c) for i, c in enumerate(logo_colors)]

LOGO_LINES = [
    "".join(
        glyph[row] + ("  " if g_idx == 0 else (" " if g_idx < len(LOGO_GLYPHS) - 1 else ""))
        for g_idx, glyph in enumerate(LOGO_GLYPHS)
    )
    for row in range(6)
]


def _render_logo() -> Text:
    """Render the logo with uniform color per alphabet, eliminating vertical cell transition lines."""
    logo = Text()
    for row in range(6):
        logo.append("  ")
        for g_idx, glyph in enumerate(LOGO_GLYPHS):
            color = logo_colors[g_idx]
            logo.append(glyph[row], style=f"bold {color}")
            if g_idx == 0:
                logo.append("  ")
            elif g_idx < len(LOGO_GLYPHS) - 1:
                logo.append(" ")
        logo.append("\n")
    return logo



# ── History & Readline Navigation ────────────────────────────────────────────

HISTORY_FILE = os.path.expanduser("~/.inquira_history")

COMMANDS_LIST = [
    "/menu",
    "/provider",
    "/model",
    "/temp",
    "/json",
    "/info",
    "/help",
    "/clear",
    "exit",
    "quit",
]

PROVIDER_OPTIONS = ["openai", "anthropic"]
MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]
TEMP_OPTIONS = ["0.0", "0.2", "0.3", "0.5", "0.7", "1.0"]


def _history_completer(text: str, state: int):
    """Context-aware Readline auto-completion for slash commands and arguments."""
    try:
        line = readline.get_line_buffer()
        parts = line.split()

        # If line has no spaces, complete top-level command
        if len(parts) == 0 or (len(parts) == 1 and not line.endswith(" ")):
            candidates = [cmd for cmd in COMMANDS_LIST if cmd.startswith(text)]
        elif parts[0] == "/provider":
            candidates = [p for p in PROVIDER_OPTIONS if p.startswith(text)]
        elif parts[0] == "/model":
            candidates = [m for m in MODEL_OPTIONS if m.startswith(text)]
        elif parts[0] in ("/temp", "/temperature"):
            candidates = [t for t in TEMP_OPTIONS if t.startswith(text)]
        else:
            candidates = [cmd for cmd in COMMANDS_LIST if cmd.startswith(text)]

        if state < len(candidates):
            return candidates[state]
    except Exception:
        pass
    return None


def _setup_readline():
    """Configure GNU Readline / libedit for command history, arrow-key browsing, and tab completion."""
    try:
        # Delimiters: only whitespace, so slash commands (/menu, /provider) are treated as single tokens
        readline.set_completer_delims(" \t\n")
        readline.set_completer(_history_completer)

        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind -e")
            readline.parse_and_bind("bind ^I rl_complete")
            readline.parse_and_bind(r"bind '\e[A' ed-prev-history")
            readline.parse_and_bind(r"bind '\e[B' ed-next-history")
        else:
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\e[A": previous-history')
            readline.parse_and_bind(r'"\e[B": next-history')
    except Exception:
        pass
    try:
        if os.path.exists(HISTORY_FILE):
            readline.read_history_file(HISTORY_FILE)
        readline.set_history_length(1000)
    except Exception:
        pass


def _save_readline():
    """Save GNU Readline history to ~/.inquira_history."""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def _record_history(text: str):
    """Ensure non-empty input is preserved in readline history."""
    if not text:
        return
    try:
        n = readline.get_current_history_length()
        if n == 0 or readline.get_history_item(n) != text:
            readline.add_history(text)
    except Exception:
        pass


def _prompt_input(prompt_label: str = "Ask Inquira ❯ ", add_to_history: bool = True) -> str:
    """
    Prompt the user using GNU Readline so that Up/Down arrow keys
    cycle through history, and Left/Right arrow keys edit the line.
    """
    if sys.stdin.isatty():
        styled_prompt = f"\001\033[1;36m\002{prompt_label}\001\033[0m\002"
        try:
            val = input(styled_prompt).strip()
            if val and add_to_history:
                _record_history(val)
                _save_readline()
            return val
        except EOFError:
            raise
    else:
        return console.input(f"[bold cyan]{prompt_label}[/] ").strip()



# ── Helpers ──────────────────────────────────────────────────────────────────

_CONFIDENCE_STYLES = {
    "high": ("● HIGH", "confidence.high"),
    "medium": ("● MEDIUM", "confidence.medium"),
    "low": ("● LOW", "confidence.low"),
}


def _confidence_text(level: str) -> Text:
    label, style = _CONFIDENCE_STYLES.get(
        level.strip().lower(), ("● UNKNOWN", "muted")
    )
    return Text(label, style=style)


def _tracing_status() -> tuple[str, str]:
    """Return (display_string, style) for the current LangSmith status."""
    if os.getenv("LANGSMITH_API_KEY"):
        project = os.getenv("LANGSMITH_PROJECT", "Inquira")
        return f"✓ Active ({project})", "success"
    return "✗ Disabled", "warning"


def _greeting() -> str:
    """Return a time-aware greeting like in Pic 2 ('Good morning.' / 'Good afternoon.')."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning."
    elif 12 <= hour < 18:
        return "Good afternoon."
    else:
        return "Good evening."


# ── Raw Key Input (for arrow-key menu) ───────────────────────────────────────


def _read_key(fd: int) -> str:
    """Read a raw key sequence directly from file descriptor without buffering."""
    try:
        b = os.read(fd, 32)
    except Exception:
        return "escape"
    if not b:
        return "escape"

    # Arrow keys (ANSI and Application cursor modes)
    if b in (b"\x1b[A", b"\x1bOA"):
        return "up"
    if b in (b"\x1b[B", b"\x1bOB"):
        return "down"
    if b in (b"\x1b[C", b"\x1bOC"):
        return "right"
    if b in (b"\x1b[D", b"\x1bOD"):
        return "left"

    # Enter / Return
    if b in (b"\r", b"\n"):
        return "enter"

    # Escape
    if b == b"\x1b":
        return "escape"

    # Quit
    if b in (b"q", b"Q"):
        return "quit"

    # Vim navigation
    if b in (b"k", b"K"):
        return "up"
    if b in (b"j", b"J"):
        return "down"

    # Ctrl+C
    if b == b"\x03":
        raise KeyboardInterrupt

    # Number keys 1-9
    try:
        s = b.decode("utf-8", errors="ignore").strip()
        if s.isdigit() and s != "0":
            return f"num_{s}"
        return s
    except Exception:
        return "unknown"


# ── Interactive Select Menu (Pic 4 Style) ────────────────────────────────────


def interactive_select(
    options: list[dict],
    title: str = "Quick Actions",
    breadcrumb: str = "/menu",
    bot=None,
) -> int | None:
    """
    Display an arrow-key navigable menu inspired by Pic 4.
    Returns the index of the selected option, or None if dismissed.

    Each option dict has:
        {"label": "Display text", "icon": "emoji", "hint": "description"}
    """
    selected = 0
    n = len(options)

    def _render():
        container = Table.grid(padding=(0, 0))
        container.add_row(Text(""))

        # Breadcrumb echo like Pic 4: "$ /links"
        prompt_line = Text()
        prompt_line.append("  $ ", style="bold bright_cyan")
        prompt_line.append(breadcrumb, style="bold bright_white")
        container.add_row(prompt_line)
        container.add_row(Text(""))

        # Title in bold warm yellow (Pic 4 "Quick Links")
        container.add_row(Text(f"  {title}", style="bold bright_yellow"))
        container.add_row(Text(""))

        # Options table with 2 columns to prevent premature wrapping
        opts_table = Table.grid(padding=(0, 2))
        opts_table.add_column(no_wrap=True)
        opts_table.add_column(no_wrap=True)

        for i, opt in enumerate(options, 1):
            is_selected = (i - 1 == selected)
            icon = opt.get("icon", "")
            label = opt.get("label", "")
            hint = opt.get("hint", "")

            col1 = Text()
            col2 = Text()

            if is_selected:
                col1.append("  ❯ ", style="bold bright_cyan")
                col1.append(f"{i}. ", style="bold bright_cyan")
                col1.append(f"{icon} ", style="bold bright_cyan")
                col1.append(label, style="bold bright_white")
                if hint:
                    col2.append(f"—  {hint}", style="bright_cyan")
            else:
                col1.append("    ", style="dim")
                col1.append(f"{i}. ", style="dim")
                col1.append(f"{icon} ", style="dim")
                col1.append(label, style="dim white")
                if hint:
                    col2.append(f"—  {hint}", style="dim")

            opts_table.add_row(col1, col2)

        container.add_row(opts_table)
        container.add_row(Text(""))
        container.add_row(
            Text(
                "  ↑↓ / jk select  ·  enter / 1-9 to pick  ·  esc back",
                style="dim italic",
            )
        )
        container.add_row(Text(""))

        # Status bar (Pic 4)
        status_text = Text()
        status_text.append("❯ ", style="bold bright_cyan")
        status_text.append("↑↓ select  ·  enter open  ·  esc cancel", style="dim white")

        status_panel = Panel(
            status_text,
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
        container.add_row(status_panel)

        # Sub-footer line (Pic 4)
        info_str = "~/inquira"
        if bot:
            info_str += f" · provider: {bot.provider} · model: {bot.model_name} · temp: {bot.temperature}"
        else:
            info_str += " · status: ready"
        container.add_row(Text(f"  {info_str}", style="dim"))
        container.add_row(Text(""))

        return container

    if not sys.stdin.isatty():
        console.print(_render())
        return 0

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except Exception:
        # Fallback if termios is unavailable or fails
        console.print(_render())
        try:
            val = _prompt_input(f"Select (1-{n}) or 'q' to cancel ❯ ", add_to_history=False)
            if val.isdigit() and 1 <= int(val) <= n:
                return int(val) - 1
        except Exception:
            pass
        return None

    try:
        tty.setcbreak(fd)
        with Live(_render(), console=console, auto_refresh=False, transient=True) as live:
            live.refresh()
            while True:
                key = _read_key(fd)
                if key in ("up", "left"):
                    selected = (selected - 1) % n
                    live.update(_render(), refresh=True)
                elif key in ("down", "right"):
                    selected = (selected + 1) % n
                    live.update(_render(), refresh=True)
                elif key == "enter":
                    return selected
                elif key in ("escape", "quit"):
                    return None
                elif key.startswith("num_"):
                    idx = int(key.split("_")[1]) - 1
                    if 0 <= idx < n:
                        return idx
                # If key is unrecognized or EOF, do nothing and wait for next event
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


# ── LangSmith Status ────────────────────────────────────────────────────────


def print_langsmith_status():
    """Print a one-line LangSmith tracing status on startup."""
    status, style = _tracing_status()
    console.print(f"  [{style}]{status}[/{style}] — LangSmith Tracing")


# ── Startup Banner (Pic 2 & Pic 3 Style) ──────────────────────────────────────


def print_banner(bot, as_json: bool = False):
    """Render the modern, appealing startup banner inspired by Datpaq."""
    console.print()
    console.print("  [bold bright_green]❯[/] [bold bright_white]inquira[/]")
    console.print()

    # ── Logo with horizontal monochromatic gradient ──
    console.print(_render_logo())

    # ── Subtitle & Greeting (Datpaq style) ──
    console.print(
        "  [dim white]AI-Powered Structured Q&A Assistant[/]  [dim]·[/]  [dim white]"
        + _greeting()
        + "[/]"
    )
    console.print()

    # ── Status pills ──
    trace_label, trace_style = _tracing_status()
    status = Text("  ")
    status.append("⚡ ", style="bright_yellow")
    status.append("provider ", style="dim")
    status.append(str(bot.provider), style="bold bright_white")
    status.append("  ·  model ", style="dim")
    status.append(str(bot.model_name), style="bold bright_white")
    status.append("  ·  temp ", style="dim")
    status.append(str(bot.temperature), style="bold bright_white")
    status.append("  ·  output ", style="dim")
    status.append("JSON" if as_json else "text", style="bold bright_white")
    status.append("  ·  langsmith ", style="dim")
    status.append(trace_label, style=trace_style)
    console.print(status)
    console.print()

    # ── Guidance hints ──
    hints = Text("  ")
    hints.append("Type a question or press ", style="dim white")
    hints.append("Enter", style="bold #c084fc")
    hints.append(" for actions menu  ·  ", style="dim white")
    hints.append("Tab", style="bold bright_white")
    hints.append(" autocomplete  ·  ", style="dim white")
    hints.append("↑↓", style="bold bright_white")
    hints.append(" browse history", style="dim white")
    console.print(hints)
    console.print()

    # ── Available commands ──
    console.print("  [bold #c084fc]Available commands:[/]")

    commands = [
        ("/menu", "interactive arrow-key command & setting picker"),
        ("/provider <name>", "switch LLM provider (openai / anthropic)"),
        ("/model <name>", "switch active model (gpt-4o-mini, claude-3-5, etc.)"),
        ("/temp <float>", "set temperature (0.0 deterministic – 1.0 creative)"),
        ("/json", "toggle structured JSON response mode"),
        ("/info", "display active session configuration"),
        ("/help", "open comprehensive user guide & options"),
        ("/clear", "clear terminal screen"),
        ("exit", "quit Inquira session"),
    ]

    cmd_table = Table.grid(padding=(0, 2))
    cmd_table.add_column(style="bold #d8b4fe", min_width=20)
    cmd_table.add_column(style="dim white")

    for cmd, desc in commands:
        cmd_table.add_row(f"    {cmd}", f"—  {desc}")

    console.print(cmd_table)
    console.print()


# ── QA Response Card ────────────────────────────────────────────────────────


def print_response(question: str, response, as_json: bool = False):
    """Render a single QA response as a styled card."""
    if as_json:
        _print_json_response(response)
        return

    # Question panel
    console.print()
    console.print(
        Panel(
            Text(question, style="bold bright_white"),
            title="🔍 Question",
            title_align="left",
            border_style="bright_blue",
            padding=(0, 2),
            expand=False,
        )
    )

    # ── Answer body ──
    answer_grid = Table.grid(padding=(0, 0))

    # Main answer text
    answer_grid.add_row(Text(response.answer, style="bright_white"))
    answer_grid.add_row(Text(""))  # spacer

    # Metadata row: confidence + sources
    meta = Table(show_header=True, box=box.ROUNDED, padding=(0, 2), expand=False)
    meta.add_column("Confidence", style="key", justify="center")
    meta.add_column("Sources Needed", style="key", justify="center")

    conf = _confidence_text(response.confidence)
    sources = (
        Text("✓ Yes", style="warning")
        if response.sources_needed
        else Text("✗ No", style="muted")
    )
    meta.add_row(conf, sources)

    answer_grid.add_row(meta)
    answer_grid.add_row(Text(""))  # spacer

    # Reasoning
    answer_grid.add_row(Text("Reasoning", style="subheader"))
    answer_grid.add_row(Text(response.reasoning, style="muted"))

    # Follow-up questions
    if response.follow_up_questions:
        answer_grid.add_row(Text(""))
        answer_grid.add_row(Text("Follow-up Questions", style="subheader"))
        for fq in response.follow_up_questions:
            answer_grid.add_row(Text(f"  ❯ {fq}", style="accent"))

    console.print(
        Panel(
            answer_grid,
            title="💡 Answer",
            title_align="left",
            border_style="bright_green",
            padding=(1, 2),
            expand=False,
        )
    )


def _print_json_response(response):
    """Print a single response as syntax-highlighted JSON."""
    json_str = response.model_dump_json(indent=2)
    console.print(
        Syntax(json_str, "json", theme="monokai", line_numbers=False)
    )


# ── Batch Responses ──────────────────────────────────────────────────────────


def print_batch_header(count: int, provider: str, model: str):
    """Print batch processing header."""
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Processing ", "muted"),
                (str(count), "accent"),
                (" questions in batch with ", "muted"),
                (provider, "info"),
                (" (", "muted"),
                (model, "info"),
                (")…", "muted"),
            ),
            border_style="bright_blue",
            padding=(0, 2),
            expand=False,
        )
    )


def print_batch_responses(questions, responses, as_json: bool = False):
    """Render a batch of QA responses."""
    if as_json:
        raw = json.dumps([r.model_dump() for r in responses], indent=2)
        console.print(
            Syntax(raw, "json", theme="monokai", line_numbers=False)
        )
        return

    for i, (q, r) in enumerate(zip(questions, responses)):
        print_response(q, r)
        if i < len(questions) - 1:
            console.print(Rule(style="dim"))


# ── Demo Headers ─────────────────────────────────────────────────────────────


def print_demo_header(name: str):
    """Print a styled demo section header."""
    console.print()
    console.print(
        Rule(f"[header]═══ {name.upper()} DEMO ═══[/]", style="bright_cyan")
    )
    console.print()


# ── Interactive Help ─────────────────────────────────────────────────────────


def print_interactive_help_rich():
    """Render the interactive help guide using Rich panels and tables."""
    # ── Asking Questions ──
    console.print()
    console.print(
        Rule("[subheader]Asking Questions[/]", style="bright_magenta")
    )
    console.print(
        "  Type your question directly at the prompt and press Enter.",
        style="muted",
    )
    console.print(
        "  The assistant responds with: Answer, Confidence, Reasoning,",
        style="muted",
    )
    console.print(
        "  Follow-up Questions, and Sources requirement.", style="muted"
    )

    # ── Commands ──
    console.print()
    console.print(
        Rule("[subheader]In-Session Commands[/]", style="bright_magenta")
    )

    cmds = [
        ("/provider <name>", "Switch between OpenAI and Anthropic", "/provider anthropic"),
        ("/model <name>", "Switch active model dynamically", "/model gpt-4o"),
        ("/temp <float>", "Adjust sampling temperature (0.0–1.0)", "/temp 0.7"),
        ("/json", "Toggle JSON / structured text output", "/json"),
        ("/info", "Display active provider, model, temp & tracing", "/info"),
        ("/clear", "Clear the terminal screen", "/clear"),
        ("/help", "Display this guide", "/help"),
        ("exit / quit / :q", "End the interactive session", "exit"),
    ]

    for cmd, desc, example in cmds:
        line = Text()
        line.append(f"  {cmd:<22}", style="bright_yellow")
        line.append(f" — {desc}", style="dim white")
        console.print(line)

    # ── CLI Cheat Sheet ──
    console.print()
    console.print(
        Rule("[subheader]CLI Cheat Sheet[/]", style="bright_magenta")
    )
    console.print(
        Syntax(
            'uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "Your question"\n'
            'uv run main.py -b "Question 1" "Question 2"\n'
            "uv run main.py --instructions",
            "bash",
            theme="monokai",
            line_numbers=False,
        )
    )
    console.print()


# ── Full Instructions (Rich) ────────────────────────────────────────────────


def print_full_instructions_rich():
    """Render the complete CLI user guide with Rich formatting."""
    console.print()
    console.print("  [bold bright_green]❯[/] [bold bright_white]inquira[/]")
    console.print()

    # Logo header
    console.print(_render_logo())
    console.print("  [bold #d8b4fe]CLI USER GUIDE[/]")

    # ── 1. Flags ──
    console.print()
    console.print(
        Rule("[subheader]1 · Which Flag Does What[/]", style="bright_magenta")
    )

    flags = [
        ("-h, --help", "Show summary help and argument options"),
        ("--instructions", "Show this complete guide"),
        ("-p, --provider NAME", "LLM provider: 'openai' or 'anthropic'"),
        ("-m, --model NAME", "Exact model identifier"),
        ("-t, --temperature NUM", "Sampling temperature (0.0–1.0, default: 0.3)"),
        ("-q, --question \"...\"", "Pass a single question explicitly"),
        ("-b, --batch \"Q1\" \"Q2\"", "Process multiple questions in parallel"),
        ("-i, --interactive", "Launch interactive chat session"),
        ("--json", "Output as formatted JSON"),
        ("--demo [NAME]", "Run demos: 'all', 'qa', 'batch', or 'error'"),
    ]

    for flag, desc in flags:
        line = Text()
        line.append(f"  {flag:<26}", style="bright_yellow")
        line.append(f" {desc}", style="dim white")
        console.print(line)

    # ── 2. How to Pass Arguments ──
    console.print()
    console.print(
        Rule(
            "[subheader]2 · How to Pass Arguments[/]",
            style="bright_magenta",
        )
    )
    console.print(
        Syntax(
            '# Single question (positional or -q flag):\n'
            'uv run main.py "What is retrieval-augmented generation?"\n'
            'uv run main.py -q "What is retrieval-augmented generation?"\n'
            '\n'
            '# Batch questions (-b flag):\n'
            'uv run main.py -b "What is Python?" "What is JavaScript?" "What is Rust?"\n'
            '\n'
            '# JSON output (--json flag):\n'
            'uv run main.py --json "What is vector indexing?"\n'
            '\n'
            '# Interactive mode (no arguments, or -i):\n'
            'uv run main.py\n'
            'uv run main.py -i',
            "bash",
            theme="monokai",
            line_numbers=False,
        )
    )

    # ── 3. Switching Providers & Models ──
    console.print()
    console.print(
        Rule(
            "[subheader]3 · Switching Providers & Models[/]",
            style="bright_magenta",
        )
    )

    console.print("  [key]Method A:[/] Command-line flags", style="value")
    console.print(
        Syntax(
            'uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "What is LangChain?"\n'
            'uv run main.py -p openai -m gpt-4o "Explain quantum superposition."\n'
            'uv run main.py -p openai -m gpt-4o-mini -t 0.7 "Brainstorm 3 project ideas."',
            "bash",
            theme="monokai",
            line_numbers=False,
        )
    )

    console.print()
    console.print(
        "  [key]Method B:[/] Live inside the interactive session",
        style="value",
    )
    console.print(
        Syntax(
            "Ask Inquira ❯ /provider anthropic\n"
            "Ask Inquira ❯ /model claude-3-5-sonnet-20241022\n"
            "Ask Inquira ❯ /temp 0.5\n"
            "Ask Inquira ❯ /json",
            "bash",
            theme="monokai",
            line_numbers=False,
        )
    )

    console.print()
    console.print(
        "  [key]Method C:[/] Persistently in your .env file",
        style="value",
    )
    console.print(
        Syntax(
            "LLM_PROVIDER=openai\n"
            "OPENAI_MODEL=gpt-4o-mini\n"
            "ANTHROPIC_MODEL=claude-3-5-sonnet-20241022\n"
            "LLM_TEMPERATURE=0.3",
            "dotenv",
            theme="monokai",
            line_numbers=False,
        )
    )

    # ── 4. Popular Models ──
    console.print()
    console.print(
        Rule(
            "[subheader]4 · Popular Models[/]", style="bright_magenta"
        )
    )

    models = [
        ("OpenAI", "gpt-4o-mini", "Fast, cost-effective — DEFAULT"),
        ("OpenAI", "gpt-4o", "Flagship high-intelligence"),
        ("OpenAI", "o3-mini", "Advanced reasoning"),
        ("Anthropic", "claude-3-5-sonnet-20241022", "Coding & analysis — DEFAULT"),
        ("Anthropic", "claude-3-5-haiku-20241022", "High speed, lightweight"),
        ("Anthropic", "claude-3-haiku-20240307", "Legacy fast model"),
    ]

    for provider, model_id, notes in models:
        line = Text()
        line.append(f"  {provider:<12}", style="bright_blue")
        line.append(f"{model_id:<32}", style="bright_yellow")
        line.append(notes, style="dim")
        console.print(line)

    # ── 5. Interactive Commands ──
    console.print()
    console.print(
        Rule(
            "[subheader]5 · Interactive Slash Commands[/]",
            style="bright_magenta",
        )
    )

    slash_cmds = [
        ("/help, help, ?", "Display interactive help & instructions"),
        ("/info", "Show active provider, model, temp & tracing"),
        ("/provider <name>", "Switch provider ('openai' or 'anthropic')"),
        ("/model <name>", "Switch model name"),
        ("/temp <float>", "Adjust sampling temperature"),
        ("/json", "Toggle JSON output mode"),
        ("/clear", "Clear the terminal screen"),
        ("exit, quit, :q", "Exit the session"),
    ]

    for cmd, desc in slash_cmds:
        line = Text()
        line.append(f"  {cmd:<22}", style="bright_yellow")
        line.append(f" — {desc}", style="dim white")
        console.print(line)

    console.print()


# ── Interactive Session ──────────────────────────────────────────────────────


_MAIN_MENU_OPTIONS = [
    {"label": "Ask a question", "icon": "💬", "hint": "type question directly"},
    {"label": "Switch provider", "icon": "🔄", "hint": "OpenAI ↔ Anthropic"},
    {"label": "Switch model", "icon": "🤖", "hint": "gpt-4o-mini, claude-3-5, etc."},
    {"label": "Set temperature", "icon": "🌡️", "hint": "0.0 deterministic – 1.0 creative"},
    {"label": "Toggle JSON output", "icon": "📋", "hint": "switch text ↔ structured JSON"},
    {"label": "Session config", "icon": "ℹ️ ", "hint": "view active provider, model & tracing"},
    {"label": "User manual", "icon": "📖", "hint": "full command reference & help"},
    {"label": "Exit session", "icon": "🚪", "hint": "quit Inquira"},
]

_PROVIDER_OPTIONS = [
    {"label": "OpenAI", "icon": "🟢", "hint": "GPT-4o, GPT-4o-mini, o3-mini"},
    {"label": "Anthropic", "icon": "🟠", "hint": "Claude 3.5 Sonnet, Claude 3.5 Haiku"},
]

_OPENAI_MODEL_OPTIONS = [
    {"label": "gpt-4o-mini", "icon": "⚡", "hint": "fast & cost-effective (Default)"},
    {"label": "gpt-4o", "icon": "🧠", "hint": "flagship high-intelligence model"},
    {"label": "o3-mini", "icon": "💡", "hint": "advanced reasoning & STEM"},
]

_ANTHROPIC_MODEL_OPTIONS = [
    {"label": "claude-3-5-sonnet-20241022", "icon": "🧠", "hint": "flagship coding & analysis (Default)"},
    {"label": "claude-3-5-haiku-20241022", "icon": "⚡", "hint": "high speed, lightweight"},
    {"label": "claude-3-haiku-20240307", "icon": "📦", "hint": "legacy fast model"},
]


def _print_info(bot, as_json: bool):
    """Render current session info as a styled panel."""
    info_grid = Table.grid(padding=(0, 2))
    info_grid.add_column(style="muted", min_width=14)
    info_grid.add_column(style="bright_white")

    info_grid.add_row("  Provider", f"  {bot.provider}")
    info_grid.add_row("  Model", f"  {bot.model_name}")
    info_grid.add_row("  Temperature", f"  {bot.temperature}")
    info_grid.add_row(
        "  Output",
        f"  {'JSON' if as_json else 'Structured Text'}",
    )
    trace_label, trace_style = _tracing_status()
    info_grid.add_row("  LangSmith", Text(f"  {trace_label}", style=trace_style))

    console.print(
        Panel(
            info_grid,
            title="[info]Session Info[/]",
            title_align="left",
            border_style="bright_blue",
            padding=(0, 1),
            expand=False,
        )
    )


def _print_setting_changed(key: str, value: str):
    """Show a brief success message for a setting change."""
    console.print(
        Text.assemble(
            ("  ✓ ", "success"),
            (key, "key"),
            (" → ", "muted"),
            (value, "accent"),
        )
    )


def _handle_provider_switch_menu(bot):
    """Arrow-key menu to pick a new provider."""
    idx = interactive_select(
        _PROVIDER_OPTIONS,
        title="Select LLM Provider",
        breadcrumb="/provider",
        bot=bot,
    )
    if idx is None:
        return
    new_prov = ["openai", "anthropic"][idx]
    try:
        old = bot.provider
        bot.update_config(provider=new_prov)
        _print_setting_changed(
            "Provider",
            f"{old} → {bot.provider}  (model: {bot.model_name})",
        )
    except Exception as e:
        console.print(
            Panel(str(e), title="⚠ Error", border_style="red", padding=(0, 2), expand=False)
        )


def _handle_model_switch_menu(bot):
    """Arrow-key menu to pick a new model."""
    if bot.provider == "openai":
        opts = list(_OPENAI_MODEL_OPTIONS)
    else:
        opts = list(_ANTHROPIC_MODEL_OPTIONS)

    # Add a custom option at the end
    opts_with_custom = opts + [
        {"label": "Custom…", "icon": "✏️ ", "hint": "type custom model identifier"}
    ]

    idx = interactive_select(
        opts_with_custom,
        title=f"Select Model ({bot.provider})",
        breadcrumb="/model",
        bot=bot,
    )
    if idx is None:
        return

    if idx == len(opts):
        # Custom model — fall back to text prompt
        console.print()
        new_model = _prompt_input("  Model name ❯ ", add_to_history=False)
        if not new_model:
            return
    else:
        new_model = opts[idx]["label"]

    try:
        old = bot.model_name
        bot.update_config(model_name=new_model)
        _print_setting_changed("Model", f"{old} → {bot.model_name}")
    except Exception as e:
        console.print(
            Panel(str(e), title="⚠ Error", border_style="red", padding=(0, 2), expand=False)
        )


def _handle_temp_menu(bot):
    """Arrow-key menu to pick a temperature."""
    temp_opts = [
        {"label": "0.0", "icon": "🧊", "hint": "deterministic / strict factual"},
        {"label": "0.2", "icon": "❄️ ", "hint": "precise & focused"},
        {"label": "0.3", "icon": "🌤️ ", "hint": "balanced — recommended default"},
        {"label": "0.5", "icon": "☀️ ", "hint": "moderately creative"},
        {"label": "0.7", "icon": "🔥", "hint": "creative & exploratory"},
        {"label": "1.0", "icon": "🌋", "hint": "maximum variance & creativity"},
        {"label": "Custom…", "icon": "✏️ ", "hint": "enter custom float value"},
    ]

    idx = interactive_select(
        temp_opts,
        title="Set Sampling Temperature (0.0 – 1.0)",
        breadcrumb="/temp",
        bot=bot,
    )
    if idx is None:
        return

    if idx == len(temp_opts) - 1:
        # Custom value
        console.print()
        val = _prompt_input("  Temperature ❯ ", add_to_history=False)
        try:
            new_temp = float(val)
        except ValueError:
            console.print("  [error]Invalid value.[/] Please enter a float.", style="error")
            return
    else:
        new_temp = float(temp_opts[idx]["label"])

    try:
        old = bot.temperature
        bot.update_config(temperature=new_temp)
        _print_setting_changed("Temperature", f"{old} → {bot.temperature}")
    except Exception as e:
        console.print(
            Panel(str(e), title="⚠ Error", border_style="red", padding=(0, 2), expand=False)
        )


def _handle_menu_choice(idx: int | None, bot, as_json: bool) -> tuple[bool, bool]:
    """
    Handle selection from _MAIN_MENU_OPTIONS.
    Returns (should_exit, as_json).
    """
    if idx is None or idx == 7:
        # Exit or dismissed
        if idx == 7:
            console.print()
            console.print(
                Panel(
                    Text("Goodbye! 👋", style="bold bright_cyan", justify="center"),
                    border_style="bright_cyan",
                    padding=(0, 4),
                    expand=False,
                )
            )
            return True, as_json
        return False, as_json

    if idx == 0:
        # Ask a question
        console.print()
        raw = _prompt_input("Ask Inquira ❯ ")
        if raw:
            with console.status("[bold cyan]Thinking…[/]", spinner="dots"):
                response = bot.ask(raw)
            print_response(raw, response, as_json=as_json)
        return False, as_json

    elif idx == 1:
        _handle_provider_switch_menu(bot)
        return False, as_json

    elif idx == 2:
        _handle_model_switch_menu(bot)
        return False, as_json

    elif idx == 3:
        _handle_temp_menu(bot)
        return False, as_json

    elif idx == 4:
        as_json = not as_json
        status = "ENABLED" if as_json else "DISABLED"
        style = "success" if as_json else "warning"
        console.print(f"  [{style}]JSON output {status}[/{style}]")
        return False, as_json

    elif idx == 5:
        _print_info(bot, as_json)
        return False, as_json

    elif idx == 6:
        print_interactive_help_rich()
        return False, as_json

    return False, as_json


def run_interactive_tui(bot, as_json: bool = False):
    """Run the Rich-powered interactive session with arrow-key menus and readline history."""
    _setup_readline()
    print_banner(bot, as_json)

    try:
        while True:
            try:
                raw = _prompt_input("Ask Inquira ❯ ")

                # Empty input or /menu opens interactive menu
                if not raw or raw.lower() in ("/menu", "menu", ":menu", "/actions"):
                    idx = interactive_select(
                        _MAIN_MENU_OPTIONS,
                        title="Quick Actions",
                        breadcrumb="/menu",
                        bot=bot,
                    )
                    should_exit, as_json = _handle_menu_choice(idx, bot, as_json)
                    if should_exit:
                        break
                    continue

                # Check for exit commands
                if raw.lower() in ("exit", "quit", ":q", "q"):
                    console.print()
                    console.print(
                        Panel(
                            Text("Goodbye! 👋", style="bold bright_cyan", justify="center"),
                            border_style="bright_cyan",
                            padding=(0, 4),
                            expand=False,
                        )
                    )
                    break

                # Help
                if raw.lower() in ("/help", "help", "?"):
                    print_interactive_help_rich()
                    continue

                # Clear
                if raw.lower() in ("/clear", "clear", "cls"):
                    console.clear()
                    print_banner(bot, as_json)
                    continue

                # Info
                if raw.lower() in ("/info", "info"):
                    _print_info(bot, as_json)
                    continue

                # JSON toggle
                if raw.lower() in ("/json", "json"):
                    as_json = not as_json
                    status = "ENABLED" if as_json else "DISABLED"
                    style = "success" if as_json else "warning"
                    console.print(f"  [{style}]JSON output {status}[/{style}]")
                    continue

                # Provider switch
                if raw.startswith("/provider"):
                    parts = raw.split(maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            old = bot.provider
                            bot.update_config(provider=parts[1].strip())
                            _print_setting_changed(
                                "Provider",
                                f"{old} → {bot.provider}  (model: {bot.model_name})",
                            )
                        except Exception as e:
                            console.print(
                                Panel(str(e), title="⚠ Error", border_style="red", expand=False)
                            )
                    else:
                        _handle_provider_switch_menu(bot)
                    continue

                # Model switch
                if raw.startswith("/model"):
                    parts = raw.split(maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            old = bot.model_name
                            bot.update_config(model_name=parts[1].strip())
                            _print_setting_changed("Model", f"{old} → {bot.model_name}")
                        except Exception as e:
                            console.print(
                                Panel(str(e), title="⚠ Error", border_style="red", expand=False)
                            )
                    else:
                        _handle_model_switch_menu(bot)
                    continue

                # Temperature switch
                if raw.startswith(("/temp", "/temperature")):
                    parts = raw.split(maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            new_temp = float(parts[1].strip())
                            old = bot.temperature
                            bot.update_config(temperature=new_temp)
                            _print_setting_changed("Temperature", f"{old} → {bot.temperature}")
                        except ValueError:
                            console.print("  [error]Invalid value.[/] Enter a float.", style="error")
                        except Exception as e:
                            console.print(
                                Panel(str(e), title="⚠ Error", border_style="red", expand=False)
                            )
                    else:
                        _handle_temp_menu(bot)
                    continue

                # Regular question
                with console.status("[bold cyan]Thinking…[/]", spinner="dots"):
                    response = bot.ask(raw)

                print_response(raw, response, as_json=as_json)

            except (KeyboardInterrupt, EOFError):
                console.print()
                console.print(
                    Panel(
                        Text("Session ended. Goodbye! 👋", style="bold bright_cyan", justify="center"),
                        border_style="bright_cyan",
                        padding=(0, 4),
                        expand=False,
                    )
                )
                break
    finally:
        _save_readline()
