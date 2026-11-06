from pursuit_game.game import Game


def main() -> None:
    Game().run()


if __name__ == "__main__":
    # Нашел ошибку ... main.py был пустым, поэтому привычный запуск проекта
    # через эту точку входа завершался без открытия игры.
    main()
