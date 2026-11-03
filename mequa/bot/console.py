"""Talk to the Mequa bot in the terminal, with no WhatsApp: python -m mequa.bot.console"""

from __future__ import annotations

import pathlib

from mequa.bot.core import BotCore, load_dataset
from mequa.utils.env import load_env_file


class ConsoleOut:
    def text(self, s):
        print(f"\nMequa: {s}")

    def menu(self, s):
        print(f"\nMequa: {s}")

    def image(self, png, caption):
        path = pathlib.Path("outputs") / "bot_map.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(png)
        print(f"\nMequa: [image saved to {path}] {caption}")


def main() -> None:
    load_env_file()
    core = BotCore(load_dataset())
    out = ConsoleOut()
    print(f"Dataset: {core.dataset.label}. Type 'help'. Ctrl+C to quit.")
    while True:
        try:
            line = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        core.handle(line, out)


if __name__ == "__main__":
    main()
