# Игра преследования и курсовая работа

Проект содержит симуляцию игры преследования, численные HJI-модели, тесты и
исходник курсовой работы.

## Запуск

Требуется Python 3.10+ с Tkinter. Для самой игры сторонние пакеты не нужны:
сетка функции цены V(x, y) лежит в `data/reduced_hji.json`. NumPy нужен только
для её пересчёта (`python -m pursuit_game.arena_policy`), NumPy и matplotlib ---
для построения графиков курсовой (`docs/coursework/make_figures.py`,
`docs/coursework/make_arena_figures.py`).

```powershell
python main.py
```

Также игру можно запустить как пакет:

```powershell
python -m pursuit_game
```

## Тесты и HJI

```powershell
python -m unittest discover -s tests -v
python -m pursuit_game.hji_solver --validate
```

Решатель редуцированной игры (полулагранжева схема, итерации по значению)
пишет результат в `data/hji_solution.json`. Игра в арене с четырьмя стенами
использует ту же редуцированную двумерную сетку `data/reduced_hji.json`
(`pursuit_game/arena_policy.py`): стены учитываются коротким прогнозом
движения с упором в стены, а состояние после прогноза оценивается по V(x, y).
Пятимерная сетка по полному состоянию арены больше не используется.

## Структура

```text
main.py                     точка запуска
pursuit_game/               игровая логика, модели и решатели
tests/                      модульные тесты
data/                       сохранённые сеточные решения HJI
docs/coursework/             исходник курсовой и инструкция сборки
docs/coursework/figures/      иллюстрации LaTeX-документа
build/coursework/             PDF, логи и прочие результаты сборки
```

Инструкции по компиляции документа находятся в
`docs/coursework/README.md`. Временные файлы и Python-кэш игнорируются Git.
