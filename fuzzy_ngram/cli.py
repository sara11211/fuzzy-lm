from termcolor import colored, cprint
from .corrector import Corrector


def print_header():
    """Prints a styled header for the CLI."""
    print()
    print(colored("  Type 'quit' to exit | 'verbose' to toggle candidate breakdown", "dark_grey"))
    print(colored("─" * 65, "dark_grey"))
    print()


def print_corrected_sentence(results: list):
    """Prints the corrected sentence with color coding."""
    parts = []
    for r in results:
        if r["unknown"]:
            parts.append(colored(r["original"], "red", attrs=["underline"]))
        elif r["changed"]:
            original = colored(r["original"], "yellow")
            arrow = colored("→", "dark_grey")
            corrected = colored(r["corrected"], "yellow", attrs=["bold"])
            parts.append(f"{original}{arrow}{corrected}")
        else:
            parts.append(colored(r["corrected"], "green"))

    print()
    print(colored("  Output:  ", "dark_grey") + " ".join(parts))
    print()


def print_candidates(results: list):
    """Prints a styled candidate breakdown for each corrected word."""
    changed = [r for r in results if r["changed"]]
    unknown = [r for r in results if r["unknown"]]

    if not changed and not unknown:
        print(colored("  No corrections needed — sentence looks good!", "green"))
        print()
        return

    print(colored("─" * 48, "dark_grey"))
    print(colored("  Candidates Breakdown", "cyan", attrs=["bold"]))
    print(colored("─" * 48, "dark_grey"))

    for r in changed:
        # Word header
        print()
        print(
            colored("  ◆ ", "cyan") +
            colored(f"'{r['original']}'", "yellow") +
            colored(" → ", "dark_grey") +
            colored(f"'{r['corrected']}'", "yellow", attrs=["bold"])
        )

        # Column headers
        print(
            colored(f"    {'WORD':<14}", "dark_grey") +
            colored(f"{'SIM':>8}", "dark_grey") +
            colored(f"{'CTX':>10}", "dark_grey") +
            colored(f"{'SCORE':>10}", "dark_grey")
        )
        print(colored("    " + "·" * 40, "dark_grey"))

        # Candidates
        for i, (word, sim, ctx, combined) in enumerate(r["candidates"]):
            if i == 0:
                # Best candidate — highlighted
                marker = colored("    ", "green")
                word_str = colored(f"{word:<14}", "green", attrs=["bold"])
                sim_str = colored(f"{sim:>8.3f}", "green")
                ctx_str = colored(f"{ctx:>10.4f}", "green")
                score_str = colored(f"{combined:>10.3f}", "green", attrs=["bold"])
            else:
                # Other candidates — dimmed
                marker = colored("    ", "dark_grey")
                word_str = colored(f"{word:<14}", "white")
                sim_str = colored(f"{sim:>8.3f}", "dark_grey")
                ctx_str = colored(f"{ctx:>10.4f}", "dark_grey")
                score_str = colored(f"{combined:>10.3f}", "dark_grey")

            print(marker + word_str + sim_str + ctx_str + score_str)

    # Show unknown words separately
    if unknown:
        print()
        print(colored("    Could not correct:", "red", attrs=["bold"]))
        for r in unknown:
            print(colored(f"    · '{r['original']}' — no confident match found", "red"))

    print()
    print(colored("─" * 48, "dark_grey"))
    print()


def print_stats(results: list):
    """Prints a short summary line after each correction."""
    total = len(results)
    changed = sum(1 for r in results if r["changed"])
    unknown = sum(1 for r in results if r["unknown"])
    correct = total - changed - unknown

    print(
        colored("  Stats: ", "dark_grey") +
        colored(f"{correct} correct  ", "green") +
        colored(f"{changed} corrected  ", "yellow") +
        colored(f"{unknown} unknown", "red")
    )
    print()


def run_cli(corrector: Corrector, verbose: bool = True):
    """
    Runs the interactive CLI loop.

    Args:
        corrector (Corrector): A ready-to-use Corrector instance.
        verbose (bool): If True, shows candidates breakdown after each correction.
    """
    print_header()

    while True:
        try:
            sentence = input(colored("  >>> ", "cyan"))
        except (EOFError, KeyboardInterrupt):
            print()
            cprint("  Goodbye!", "cyan")
            break

        # Commands
        if sentence.strip().lower() == "quit":
            cprint("  Goodbye!", "cyan")
            break

        if sentence.strip().lower() == "verbose":
            verbose = not verbose
            state = "ON" if verbose else "OFF"
            print(colored(f"  Verbose mode: {state}", "cyan"))
            print()
            continue

        if not sentence.strip():
            continue

        results = corrector.correct_sentence(sentence)

        print_corrected_sentence(results)
        print_stats(results)

        if verbose:
            print_candidates(results)