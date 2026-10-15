"""
Inquira TUI — Rich-powered terminal user interface.

All display logic lives here. The SmartQABot class in main.py stays
purely functional; this module wraps its output in styled Rich panels,
tables, spinners, and prompts.
"""

import json
import os

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
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


# ── LangSmith Status ────────────────────────────────────────────────────────


def print_langsmith_status():
    """Print a one-line LangSmith tracing status on startup."""
    status, style = _tracing_status()
    console.print(f"  [{style}]{status}[/{style}] — LangSmith Tracing")


# ── Startup Banner ───────────────────────────────────────────────────────────


def print_banner(bot, as_json: bool = False):
    """Render the interactive session startup banner."""
    # ── Title ──
    title_text = Text()
    title_text.append("✦  ", style="bright_yellow")
    title_text.append("I N Q U I R A", style="bold bright_cyan")
    title_text.append("  ✦", style="bright_yellow")

    subtitle = Text(
        "AI-Powered Structured Q&A Assistant", style="bright_magenta"
    )

    # ── Config table ──
    cfg = Table(
        show_header=False,
        box=box.SIMPLE_HEAVY,
        padding=(0, 2),
        expand=True,
    )
    cfg.add_column("Key", style="key", ratio=1)
    cfg.add_column("Value", style="value", ratio=3)

    cfg.add_row("Provider", bot.provider)
    cfg.add_row("Model", bot.model_name)
    cfg.add_row("Temperature", str(bot.temperature))
    cfg.add_row(
        "Output",
        "JSON (schema formatted)" if as_json else "Structured Text",
    )
    trace_label, trace_style = _tracing_status()
    cfg.add_row("LangSmith", Text(trace_label, style=trace_style))

    # ── Quick commands ──
    cmds = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    cmds.add_column(style="accent")
    cmds.add_column(style="muted")
    cmds.add_column(style="accent")
    cmds.add_column(style="muted")

    cmds.add_row(
        "/provider <name>", "switch provider",
        "/model <name>", "switch model",
    )
    cmds.add_row(
        "/temp <float>", "set temperature",
        "/json", "toggle JSON output",
    )
    cmds.add_row(
        "/info", "show config",
        "/help", "full manual",
    )
    cmds.add_row(
        "/clear", "clear screen",
        "exit", "quit session",
    )

    # ── Compose ──
    inner = Table.grid(padding=(1, 0))
    inner.add_row(Text.assemble(title_text, justify="center"))
    inner.add_row(Text.assemble(subtitle, justify="center"))
    inner.add_row(cfg)
    inner.add_row(Rule(style="bright_cyan"))
    inner.add_row(
        Text("Quick Commands", style="bold bright_cyan", justify="center")
    )
    inner.add_row(cmds)

    console.print()
    console.print(
        Panel(
            inner,
            border_style="bright_cyan",
            padding=(1, 3),
            expand=False,
        )
    )


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
                (")...", "muted"),
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

    # ── Slash Commands Table ──
    console.print()
    console.print(
        Rule("[subheader]In-Session Commands[/]", style="bright_magenta")
    )

    cmd_table = Table(
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
        title_style="subheader",
    )
    cmd_table.add_column("Command", style="accent", min_width=24)
    cmd_table.add_column("Description", style="value")
    cmd_table.add_column("Example", style="muted")

    cmd_table.add_row(
        "/provider <name>",
        "Switch between OpenAI and Anthropic",
        "/provider anthropic",
    )
    cmd_table.add_row(
        "/model <name>",
        "Switch active model dynamically",
        "/model gpt-4o",
    )
    cmd_table.add_row(
        "/temp <float>",
        "Adjust sampling temperature (0.0–1.0)",
        "/temp 0.7",
    )
    cmd_table.add_row(
        "/json", "Toggle JSON / structured text output", "/json"
    )
    cmd_table.add_row(
        "/info",
        "Display active provider, model, temp & tracing",
        "/info",
    )
    cmd_table.add_row("/clear", "Clear the terminal screen", "/clear")
    cmd_table.add_row(
        "/help", "Display this guide", "/help"
    )
    cmd_table.add_row(
        "exit / quit / :q", "End the interactive session", "exit"
    )

    console.print(cmd_table)

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
    console.print(
        Panel(
            Text.assemble(
                ("✦  ", "bright_yellow"),
                ("INQUIRA CLI USER GUIDE", "bold bright_cyan"),
                ("  ✦", "bright_yellow"),
                justify="center",
            ),
            border_style="bright_cyan",
            padding=(0, 4),
            expand=False,
        )
    )

    # ── 1. Flags ──
    console.print()
    console.print(
        Rule("[subheader]1 · Which Flag Does What[/]", style="bright_magenta")
    )

    flags_table = Table(
        box=box.ROUNDED, padding=(0, 2), expand=False
    )
    flags_table.add_column("Flag", style="accent", min_width=10)
    flags_table.add_column("Long Form", style="accent", min_width=18)
    flags_table.add_column("Description", style="value")

    flags_table.add_row("-h", "--help", "Show summary help and argument options")
    flags_table.add_row("", "--instructions", "Show this complete guide")
    flags_table.add_row("-p", "--provider NAME", "LLM provider: 'openai' or 'anthropic'")
    flags_table.add_row("-m", "--model NAME", "Exact model identifier")
    flags_table.add_row("-t", "--temperature NUM", "Sampling temperature (0.0–1.0, default: 0.3)")
    flags_table.add_row("-q", "--question \"...\"", "Pass a single question explicitly")
    flags_table.add_row("-b", "--batch \"Q1\" \"Q2\"", "Process multiple questions in parallel")
    flags_table.add_row("-i", "--interactive", "Launch interactive chat session")
    flags_table.add_row("", "--json", "Output as formatted JSON")
    flags_table.add_row("", "--demo [NAME]", "Run demos: 'all', 'qa', 'batch', or 'error'")

    console.print(flags_table)

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
            'uv run main.py -b "Question 1" "Question 2" --json\n'
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
            '# Query Anthropic Claude:\n'
            'uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "What is LangChain?"\n'
            '\n'
            '# Query OpenAI GPT-4o:\n'
            'uv run main.py -p openai -m gpt-4o "Explain quantum superposition."\n'
            '\n'
            '# Custom temperature for creative queries:\n'
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
            "# Start session, then use slash commands:\n"
            "Ask Inquira > /provider anthropic\n"
            "Ask Inquira > /model claude-3-5-sonnet-20241022\n"
            "Ask Inquira > /temp 0.5\n"
            "Ask Inquira > /json\n"
            "Ask Inquira > /info",
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

    models_table = Table(box=box.ROUNDED, padding=(0, 2), expand=False)
    models_table.add_column("Provider", style="info", min_width=12)
    models_table.add_column("Model ID", style="accent", min_width=30)
    models_table.add_column("Notes", style="muted")

    models_table.add_row("OpenAI", "gpt-4o-mini", "Fast, cost-effective — DEFAULT")
    models_table.add_row("OpenAI", "gpt-4o", "Flagship high-intelligence")
    models_table.add_row("OpenAI", "o3-mini", "Advanced reasoning")
    models_table.add_row("Anthropic", "claude-3-5-sonnet-20241022", "Coding & analysis — DEFAULT")
    models_table.add_row("Anthropic", "claude-3-5-haiku-20241022", "High speed, lightweight")
    models_table.add_row("Anthropic", "claude-3-haiku-20240307", "Legacy fast model")

    console.print(models_table)

    # ── 5. Interactive Slash Commands ──
    console.print()
    console.print(
        Rule(
            "[subheader]5 · Interactive Slash Commands[/]",
            style="bright_magenta",
        )
    )

    slash_table = Table(box=box.ROUNDED, padding=(0, 2), expand=False)
    slash_table.add_column("Command", style="accent", min_width=22)
    slash_table.add_column("Description", style="value")

    slash_table.add_row("/help, help, ?", "Display interactive help & instructions")
    slash_table.add_row("/info", "Show active provider, model, temp & tracing")
    slash_table.add_row("/provider <name>", "Switch provider ('openai' or 'anthropic')")
    slash_table.add_row("/model <name>", "Switch model name")
    slash_table.add_row("/temp <float>", "Adjust sampling temperature")
    slash_table.add_row("/json", "Toggle JSON output mode")
    slash_table.add_row("/clear", "Clear the terminal screen")
    slash_table.add_row("exit, quit, :q", "Exit the session")

    console.print(slash_table)
    console.print()


# ── Interactive Session ──────────────────────────────────────────────────────


def _print_info(bot, as_json: bool):
    """Render current session info as a styled panel."""
    info_table = Table(
        show_header=False, box=box.SIMPLE_HEAVY, padding=(0, 2), expand=False
    )
    info_table.add_column("Key", style="key")
    info_table.add_column("Value", style="value")

    info_table.add_row("Provider", bot.provider)
    info_table.add_row("Model", bot.model_name)
    info_table.add_row("Temperature", str(bot.temperature))
    info_table.add_row(
        "Output",
        "JSON" if as_json else "Structured Text",
    )
    trace_label, trace_style = _tracing_status()
    info_table.add_row("LangSmith", Text(trace_label, style=trace_style))

    console.print(
        Panel(
            info_table,
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


def run_interactive_tui(bot, as_json: bool = False):
    """Run the Rich-powered interactive session."""
    print_banner(bot, as_json)

    while True:
        try:
            console.print()
            raw = console.input("[bold cyan]Ask Inquira ❯[/] ").strip()
            if not raw:
                continue

            # ── Exit ──
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

            # ── Help ──
            if raw.lower() in ("/help", "help", "?"):
                print_interactive_help_rich()
                continue

            # ── Clear ──
            if raw.lower() in ("/clear", "clear", "cls"):
                console.clear()
                continue

            # ── Info ──
            if raw.lower() in ("/info", "info"):
                _print_info(bot, as_json)
                continue

            # ── Toggle JSON ──
            if raw.lower() in ("/json", "json"):
                as_json = not as_json
                status = "ENABLED" if as_json else "DISABLED"
                style = "success" if as_json else "warning"
                console.print(f"  [{style}]JSON output {status}[/{style}]")
                continue

            # ── /provider ──
            if raw.startswith("/provider"):
                parts = raw.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print(
                        "  [warning]Usage:[/] /provider <openai|anthropic>"
                    )
                    console.print(f"  Current: [accent]{bot.provider}[/]")
                    continue
                new_prov = parts[1].strip().lower()
                try:
                    old = bot.provider
                    bot.update_config(provider=new_prov)
                    _print_setting_changed(
                        "Provider",
                        f"{old} → {bot.provider}  (model: {bot.model_name})",
                    )
                except Exception as e:
                    console.print(
                        Panel(
                            str(e),
                            title="⚠ Error",
                            border_style="red",
                            padding=(0, 2),
                            expand=False,
                        )
                    )
                continue

            # ── /model ──
            if raw.startswith("/model"):
                parts = raw.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print(
                        "  [warning]Usage:[/] /model <model_name>"
                    )
                    console.print(
                        f"  Current: [accent]{bot.model_name}[/]"
                    )
                    continue
                new_model = parts[1].strip()
                try:
                    old = bot.model_name
                    bot.update_config(model_name=new_model)
                    _print_setting_changed("Model", f"{old} → {bot.model_name}")
                except Exception as e:
                    console.print(
                        Panel(
                            str(e),
                            title="⚠ Error",
                            border_style="red",
                            padding=(0, 2),
                            expand=False,
                        )
                    )
                continue

            # ── /temp ──
            if raw.startswith(("/temp", "/temperature")):
                parts = raw.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    console.print(
                        "  [warning]Usage:[/] /temp <float>  (e.g. 0.3, 0.7)"
                    )
                    console.print(
                        f"  Current: [accent]{bot.temperature}[/]"
                    )
                    continue
                try:
                    new_temp = float(parts[1].strip())
                    old = bot.temperature
                    bot.update_config(temperature=new_temp)
                    _print_setting_changed(
                        "Temperature", f"{old} → {bot.temperature}"
                    )
                except ValueError:
                    console.print(
                        "  [error]Invalid value.[/] Please enter a float (e.g. 0.3)"
                    )
                except Exception as e:
                    console.print(
                        Panel(
                            str(e),
                            title="⚠ Error",
                            border_style="red",
                            padding=(0, 2),
                            expand=False,
                        )
                    )
                continue

            # ── Question ──
            with console.status(
                "[bold cyan]Thinking…[/]", spinner="dots"
            ):
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
