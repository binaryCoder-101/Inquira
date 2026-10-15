import argparse
import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from pydantic import BaseModel, Field

load_dotenv()

# LangSmith Configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Inquira")
    print(f"LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")
else:
    print("LangSmith is not configured.")


# Pydantic Schema
class QAResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    confidence: str = Field(description="Confidence level: high, medium, low")
    reasoning: str = Field(description="Reasoning behind the answer")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions related to the topic", 
        default_factory=list
    )
    sources_needed: bool = Field(
        description="Whether sources are needed to answer the question",
        default=False
    )


# Bot implementation
class SmartQABot:
    def __init__(
        self, 
        provider: Optional[str] = None,
        model_name: Optional[str] = None, 
        temperature: Optional[float] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()

        if temperature is None:
            temp_env = os.getenv("LLM_TEMPERATURE")
            self.temperature = float(temp_env) if temp_env else 0.3
        else:
            self.temperature = float(temperature)

        if self.provider == "openai":
            self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "anthropic":
            self.model_name = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        else:
            self.model_name = model_name or "default"

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    """
                    You are a knowledgeable Q&A assistant.

                    Your guidelines:
                    1. Answer the question accurately and concisely.
                    2. Be honest about uncertainity - set confidence to 'low' if unsure.
                    3. Provide clear reasoning for your answers.
                    4. Suggest relevant follow-up questions if applicable.
                    5. Determine if external sources are needed to answer the question.
                    
                    Always respond with accurate, helpful information
                    """,
                ),
                ("human", "Question: {question}"),
            ]
        )
        self._rebuild_chain()

    def _rebuild_chain(self):
        """Construct or re-construct the LangChain execution chain based on current config."""
        if self.provider == "openai":
            base_llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif self.provider == "anthropic":
            base_llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
            )
        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. Supported providers are: 'openai', 'anthropic'."
            )

        self.model = base_llm.with_structured_output(QAResponse)
        self.chain = self.prompt | self.model

    def update_config(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Dynamically update bot settings during an interactive session."""
        if provider:
            p = provider.strip().lower()
            if p not in ("openai", "anthropic"):
                raise ValueError(f"Unsupported provider '{provider}'. Must be 'openai' or 'anthropic'.")
            self.provider = p
            if not model_name:
                self.model_name = (
                    os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    if p == "openai"
                    else os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                )

        if model_name:
            self.model_name = model_name.strip()

        if temperature is not None:
            self.temperature = float(temperature)

        self._rebuild_chain()

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            # return a graceful error response
            return QAResponse(
                answer="I am sorry, I am not able to answer your question at the moment.",
                confidence="low",
                reasoning=f"Error: {str(e)}",
                follow_up_questions=["Could you please try again later?"],
                sources_needed=False
            )
    
    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        try:
            responses = self.chain.batch([{"question": question} for question in questions])
            return responses
        except Exception as e:
            # return a graceful error response
            return [QAResponse(
                answer="I am sorry, I am not able to answer your question at the moment.",
                confidence="low",
                reasoning=f"Error: {str(e)}",
                follow_up_questions=["Could you please try again later?"],
                sources_needed=False
            ) for _ in questions]


def print_qa_response(question: str, response: QAResponse, as_json: bool = False):
    """Format and print a single QA response."""
    if as_json:
        print(response.model_dump_json(indent=2))
        return

    print(f"\nQuestion: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    if response.follow_up_questions:
        print("Follow-up questions:")
        for q in response.follow_up_questions:
            print(f"  - {q}")
    print(f"Sources needed: {response.sources_needed}")


def print_full_instructions():
    """Print the complete guide for flags, arguments, providers, models, and interactive usage."""
    print("""
================================================================================
                         INQUIRA CLI USER GUIDE
================================================================================

1. WHICH FLAG DOES WHAT:
--------------------------------------------------------------------------------
  -h, --help            Show summary help and argument options.
  --instructions        Show this complete guide on flags, models, and usage.
  -p, --provider NAME   Select LLM provider: 'openai' or 'anthropic'.
                        Default: $LLM_PROVIDER or 'openai'.
  -m, --model NAME      Specify the exact model identifier.
                        Default for OpenAI   : 'gpt-4o-mini'
                        Default for Anthropic: 'claude-3-5-sonnet-20241022'
  -t, --temperature NUM Sampling temperature between 0.0 (deterministic) and 1.0 (creative).
                        Default: 0.3 or $LLM_TEMPERATURE.
  -q, --question "..."  Pass a single question string explicitly.
  -b, --batch "Q1" "Q2" Pass multiple questions to process in parallel/batch.
  -i, --interactive     Launch the interactive chat session (default if no flags).
  --json                Output responses as formatted JSON instead of human text.
  --demo [NAME]         Run built-in test suites: 'all', 'qa', 'batch', or 'error'.

2. HOW TO PASS ARGUMENTS:
--------------------------------------------------------------------------------
  • Ask a single question directly (positional or -q flag):
      uv run main.py "What is retrieval-augmented generation?"
      uv run main.py -q "What is retrieval-augmented generation?"

  • Ask multiple questions in batch (-b flag):
      uv run main.py -b "What is Python?" "What is JavaScript?" "What is Rust?"

  • Get machine-readable JSON output (--json flag):
      uv run main.py --json "What is vector indexing?"
      uv run main.py -b "Question 1" "Question 2" --json

  • Start interactive mode (run with no arguments, or with -i):
      uv run main.py
      uv run main.py -i

3. HOW TO SWITCH PROVIDERS AND MODELS:
--------------------------------------------------------------------------------
  Method A: From the command line using -p and -m flags
    # Query Anthropic Claude:
    uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "What is LangChain?"

    # Query OpenAI GPT-4o:
    uv run main.py -p openai -m gpt-4o "Explain quantum superposition."

    # Use a custom temperature (e.g. creative ideation):
    uv run main.py -p openai -m gpt-4o-mini -t 0.7 "Brainstorm 3 project ideas."

  Method B: Live inside the interactive session (no restart needed)
    Run 'uv run main.py', then type slash commands directly at the prompt:
      Ask Inquira > /provider anthropic
      Ask Inquira > /model claude-3-5-sonnet-20241022
      Ask Inquira > /temp 0.5
      Ask Inquira > /json
      Ask Inquira > /info

  Method C: Persistently in your .env file
    Add or edit these keys in your .env file:
      LLM_PROVIDER=openai                       # 'openai' or 'anthropic'
      OPENAI_MODEL=gpt-4o-mini                  # Default OpenAI model
      ANTHROPIC_MODEL=claude-3-5-sonnet-20241022# Default Anthropic model
      LLM_TEMPERATURE=0.3                       # Default temperature

4. POPULAR MODELS FOR EACH PROVIDER:
--------------------------------------------------------------------------------
  OpenAI (-p openai):
    • gpt-4o-mini                 (Fast, lightweight, cost-effective - DEFAULT)
    • gpt-4o                      (Flagship high-intelligence model)
    • o3-mini                     (Advanced reasoning model)

  Anthropic (-p anthropic):
    • claude-3-5-sonnet-20241022  (Smartest Anthropic model, coding & analysis - DEFAULT)
    • claude-3-5-haiku-20241022   (High-speed Anthropic model)
    • claude-3-haiku-20240307     (Legacy fast model)

5. INTERACTIVE SESSION SLASH COMMANDS:
--------------------------------------------------------------------------------
  /help, help, ?        Display interactive help and instructions
  /info                 Show active provider, model, temperature, and tracing status
  /provider <name>      Switch provider ('openai' or 'anthropic')
  /model <name>         Switch model name (e.g. gpt-4o, claude-3-5-sonnet-20241022)
  /temp <float>         Adjust sampling temperature (e.g. /temp 0.7)
  /json                 Toggle JSON output mode on/off
  /clear                Clear the terminal screen
  exit, quit, :q        Exit the session
================================================================================
""")


def print_interactive_help():
    """Print complete instructions for interactive mode."""
    print("""
--------------------------------------------------------------------------------
  INQUIRA INTERACTIVE HELP & INSTRUCTIONS
--------------------------------------------------------------------------------
  Asking Questions:
    • Type your question directly at the prompt and press Enter.
    • The assistant responds with structured data: Answer, Confidence,
      Reasoning, relevant Follow-up questions, and Sources requirement.

  How to Switch Models & Providers in this Session:
    /provider <name>      Switch between 'openai' and 'anthropic'
                          Example: /provider anthropic
                          Example: /provider openai
    /model <name>         Switch model name dynamically
                          Example: /model gpt-4o
                          Example: /model claude-3-5-sonnet-20241022
                          Example: /model claude-3-haiku-20240307
    /temp <float>         Adjust sampling temperature (0.0 to 1.0)
                          Example: /temp 0.7 (more creative)
                          Example: /temp 0.2 (more factual & strict)

  Other In-Session Commands:
    /json                 Toggle JSON output mode on/off
    /info                 Display current active provider, model, temperature & tracing
    /clear, clear, cls    Clear the terminal screen
    /help, help, ?        Display this guide
    exit, quit, :q        Exit the interactive session

  CLI Arguments Cheat-Sheet (when launching from terminal):
    uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "Your question"
    uv run main.py -b "Question 1" "Question 2"
    uv run main.py --instructions
--------------------------------------------------------------------------------
""")


def run_interactive(bot: SmartQABot, as_json: bool = False):
    """Run an interactive CLI session with full user instructions and in-session commands."""
    tracing_status = (
        f"Active ({os.getenv('LANGSMITH_PROJECT', 'Inquira')})"
        if os.getenv("LANGSMITH_API_KEY")
        else "Disabled"
    )

    print("\n" + "=" * 76)
    print("  INQUIRA - AI-Powered Structured Q&A Assistant")
    print("=" * 76)
    print(f"  Active Provider : {bot.provider}")
    print(f"  Active Model    : {bot.model_name}")
    print(f"  Temperature     : {bot.temperature}")
    print(f"  Output Format   : {'JSON (schema formatted)' if as_json else 'Human-readable structured text'}")
    print(f"  Observability   : LangSmith {tracing_status}")
    print("-" * 76)
    print("  HOW TO INTERACT & SWITCH CONFIGURATION:")
    print("  • Ask Question   : Type your question directly and press Enter.")
    print("  • Change Provider: /provider <openai|anthropic>  (e.g., /provider anthropic)")
    print("  • Change Model   : /model <model_name>          (e.g., /model gpt-4o)")
    print("  • Change Temp    : /temp <0.0-1.0>              (e.g., /temp 0.7)")
    print("  • Toggle JSON    : /json                        (switch Text <-> JSON output)")
    print("  • Check Settings : /info                        (view active config)")
    print("  • Full Manual    : /help                        (view detailed guide)")
    print("  • Exit Session   : exit, quit, or Ctrl+C")
    print("=" * 76)

    while True:
        try:
            raw_input = input("\nAsk Inquira > ").strip()
            if not raw_input:
                continue

            # Exit commands
            if raw_input.lower() in ("exit", "quit", ":q", "q"):
                print("Goodbye!")
                break

            # Help commands
            if raw_input.lower() in ("/help", "help", "?"):
                print_interactive_help()
                continue

            # Clear screen
            if raw_input.lower() in ("/clear", "clear", "cls"):
                os.system("cls" if os.name == "nt" else "clear")
                continue

            # Session info
            if raw_input.lower() in ("/info", "info"):
                print("\n[Current Session Info]")
                print(f"  Provider    : {bot.provider}")
                print(f"  Model       : {bot.model_name}")
                print(f"  Temperature : {bot.temperature}")
                print(f"  Output Mode : {'JSON' if as_json else 'Structured Text'}")
                print(f"  LangSmith   : {tracing_status}")
                continue

            # Toggle JSON format
            if raw_input.lower() in ("/json", "json"):
                as_json = not as_json
                print(f"[Setting Updated] JSON output mode: {'ENABLED' if as_json else 'DISABLED'}")
                continue

            # /provider command
            if raw_input.startswith("/provider"):
                parts = raw_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("[Usage] /provider <openai|anthropic>")
                    print(f"  Current provider: {bot.provider}")
                    continue
                new_prov = parts[1].strip().lower()
                try:
                    bot.update_config(provider=new_prov)
                    print(f"[Setting Updated] Provider switched to '{bot.provider}' (model: '{bot.model_name}').")
                except Exception as e:
                    print(f"[Error] Failed to switch provider: {e}")
                continue

            # /model command
            if raw_input.startswith("/model"):
                parts = raw_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("[Usage] /model <model_name>")
                    print(f"  Current model: {bot.model_name}")
                    continue
                new_model = parts[1].strip()
                try:
                    bot.update_config(model_name=new_model)
                    print(f"[Setting Updated] Model switched to '{bot.model_name}'.")
                except Exception as e:
                    print(f"[Error] Failed to switch model: {e}")
                continue

            # /temp command
            if raw_input.startswith(("/temp", "/temperature")):
                parts = raw_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("[Usage] /temp <float> (e.g. 0.3, 0.7)")
                    print(f"  Current temperature: {bot.temperature}")
                    continue
                try:
                    new_temp = float(parts[1].strip())
                    bot.update_config(temperature=new_temp)
                    print(f"[Setting Updated] Temperature set to {bot.temperature}.")
                except ValueError:
                    print("[Error] Invalid temperature value. Please enter a float (e.g., 0.3).")
                except Exception as e:
                    print(f"[Error] Failed to update temperature: {e}")
                continue

            response = bot.ask(raw_input)
            print_qa_response(raw_input, response, as_json=as_json)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


@traceable(name="qa_demo", run_type="chain")
def demo_qa_bot(bot: Optional[SmartQABot] = None):
    bot = bot or SmartQABot()
    print("=== Q&A Bot Demo ===\n")
    
    # Example 1: Single question
    print("Question 1: What is retrieval-augmented generation?")
    response = bot.ask("What is retrieval-augmented generation?")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Follow-up questions: {response.follow_up_questions}")
    print(f"Sources needed: {response.sources_needed}")
    print("\n" + "="*40 + "\n")
    
    # Example 2: Batch questions
    print("Question 2: What is LangChain?")
    print("Question 3: How does LangSmith help with observability?")
    
    questions = [
        "What is LangChain?",
        "How does LangSmith help with observability?",
    ]
    
    responses = bot.ask_batch(questions)
    
    for i, response in enumerate(responses, start=2):
        print(f"\nQuestion {i}: {questions[i-2]}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up questions: {response.follow_up_questions}")
        print(f"Sources needed: {response.sources_needed}")


@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling(bot: Optional[SmartQABot] = None):
    """Demonstrate error handling."""
    bot = bot or SmartQABot()

    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test with a very long question (edge case)
    long_question = "What is " + "very " * 100 + "important?"

    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")


@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing(bot: Optional[SmartQABot] = None):
    """Demonstrate batch processing."""
    bot = bot or SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="inquira",
        description="""
================================================================================
  INQUIRA - AI-Powered Structured Q&A Assistant
================================================================================
  Inquira is an extensible AI question-answering application built with LangChain,
  Pydantic structured output validation, and LangSmith observability tracing.
  
  It queries supported LLM providers (OpenAI, Anthropic) and guarantees responses
  conform to a validated schema:
    • answer              : Clear, concise natural language answer
    • confidence          : LLM confidence evaluation ('high', 'medium', 'low')
    • reasoning           : Explanation of the logic behind the answer
    • follow_up_questions : Suggested relevant follow-up inquiries
    • sources_needed      : Boolean flag indicating if external citations are needed
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
================================================================================
1. WHICH FLAG DOES WHAT:
================================================================================
  -h, --help            Show this help summary with argument options and exit.
  --instructions        Print the complete in-depth user guide and manual.
  -p, --provider NAME   Select LLM provider: 'openai' or 'anthropic'.
                        (default: $LLM_PROVIDER or 'openai')
  -m, --model NAME      Specify the model identifier.
                        (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
  -t, --temperature NUM Sampling temperature between 0.0 and 2.0.
                        (0.0 = strict/factual, 0.7 = creative, default: 0.3)
  -q, --question "..."  Explicit single question flag (alternative to positional argument).
  -b, --batch "Q1" "Q2" Process multiple questions concurrently in batch.
  -i, --interactive     Force launch interactive Q&A REPL session.
  --json                Output response as formatted JSON instead of structured text.
  --demo [NAME]         Run built-in test suites: 'all', 'qa', 'batch', or 'error'.

================================================================================
2. HOW TO PASS ARGUMENTS:
================================================================================
  • Ask a single question directly:
      $ uv run main.py "What is retrieval-augmented generation?"
      $ uv run main.py -q "Explain the difference between SQL and NoSQL."

  • Ask multiple questions in batch:
      $ uv run main.py -b "What is Python?" "What is JavaScript?" "What is Rust?"

  • Output formatted JSON (useful for scripts / pipelines):
      $ uv run main.py --json "What is vector indexing?"
      $ uv run main.py -b "What is Docker?" "What is Kubernetes?" --json

  • Start interactive mode (default when run without arguments):
      $ uv run main.py
      $ uv run main.py -i

================================================================================
3. HOW TO SWITCH PROVIDERS AND MODELS:
================================================================================
  Method A: From the command line using -p and -m flags
    # Query Anthropic Claude:
    $ uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "What is LangChain?"
    
    # Query OpenAI GPT-4o:
    $ uv run main.py -p openai -m gpt-4o "Explain quantum superposition."

    # Use higher temperature for creative queries:
    $ uv run main.py -p openai -m gpt-4o-mini -t 0.7 "Brainstorm 3 project ideas."

  Method B: Live inside the interactive session (no restart needed)
    Run '$ uv run main.py', then type slash commands directly at the prompt:
      Ask Inquira > /provider anthropic
      Ask Inquira > /model claude-3-5-sonnet-20241022
      Ask Inquira > /temp 0.5
      Ask Inquira > /json
      Ask Inquira > /info

  Method C: Persistently in your .env file
    Add or edit these keys in your .env file:
      LLM_PROVIDER=openai                       # 'openai' or 'anthropic'
      OPENAI_MODEL=gpt-4o-mini                  # Default OpenAI model
      ANTHROPIC_MODEL=claude-3-5-sonnet-20241022# Default Anthropic model
      LLM_TEMPERATURE=0.3                       # Default temperature

================================================================================
4. POPULAR MODEL IDENTIFIERS:
================================================================================
  OpenAI (-p openai):
    • gpt-4o-mini                 (Fast, cost-effective - DEFAULT)
    • gpt-4o                      (Flagship high intelligence)
    • o3-mini                     (Advanced reasoning model)

  Anthropic (-p anthropic):
    • claude-3-5-sonnet-20241022  (Coding & deep analysis - DEFAULT)
    • claude-3-5-haiku-20241022   (High speed, lightweight)
    • claude-3-haiku-20240307     (Legacy fast model)

================================================================================
5. INTERACTIVE LIVE SLASH COMMANDS:
================================================================================
  /help, help, ?        Display interactive help & command instructions
  /info                 Show current active provider, model, temperature & tracing
  /provider <name>      Switch provider ('openai' or 'anthropic')
  /model <name>         Switch model name (e.g. gpt-4o, claude-3-5-sonnet-20241022)
  /temp <float>         Adjust sampling temperature (e.g. /temp 0.7)
  /json                 Toggle JSON output mode on/off
  /clear                Clear the terminal screen
  exit, quit, :q        Exit the session
================================================================================
        """
    )
    parser.add_argument(
        "positional_question",
        nargs="*",
        help="Question to ask in single-query mode (e.g. 'What is RAG?'). If omitted, launches interactive mode."
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Explicit single question argument (alternative to positional argument)"
    )
    parser.add_argument(
        "-b", "--batch",
        nargs="+",
        metavar="QUESTION",
        help="List of questions to process in batch (e.g. -b 'Q1' 'Q2' 'Q3')"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force launch interactive Q&A session even if a question is provided"
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider: 'openai' or 'anthropic' (default: $LLM_PROVIDER or 'openai')"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'gpt-4o-mini', 'claude-3-5-sonnet-20241022')"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=None,
        help="Sampling temperature between 0.0 and 2.0 (default: 0.3 or $LLM_TEMPERATURE)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output responses as formatted JSON string instead of structured text"
    )
    parser.add_argument(
        "--demo",
        nargs="?",
        const="all",
        choices=["all", "qa", "batch", "error"],
        help="Run built-in demo suite: qa, batch, error, or all (default: all)"
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Show the complete guide on flags, models, providers, and argument passing"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    single_question = args.question or (" ".join(args.positional_question).strip() if args.positional_question else "")

    # Display full instructions if requested
    if args.instructions or single_question.lower() in ("instructions", "guide", "--instructions", "-instructions", "help", "?", "--help", "-h"):
        print_full_instructions()
        return

    bot = SmartQABot(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
    )

    try:
        if args.demo:
            if args.demo in ("qa", "all"):
                demo_qa_bot(bot)
            if args.demo in ("batch", "all"):
                demo_batch_processing(bot)
            if args.demo in ("error", "all"):
                demo_error_handling(bot)
        elif args.batch:
            print(f"Processing {len(args.batch)} questions in batch with {bot.provider} ({bot.model_name})...\n")
            responses = bot.ask_batch(args.batch)
            if args.json:
                print(json.dumps([r.model_dump() for r in responses], indent=2))
            else:
                for q, r in zip(args.batch, responses):
                    print_qa_response(q, r)
                    print("\n" + "-" * 40)
        else:
            if single_question and not args.interactive:
                response = bot.ask(single_question)
                print_qa_response(single_question, response, as_json=args.json)
            else:
                run_interactive(bot, as_json=args.json)
    finally:
        try:
            Client().flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()


