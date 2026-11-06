"""Run the graphical pursuit game with ``python -m pursuit_game``."""

from .game import Game


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
