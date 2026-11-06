# Курсовая работа

Исходник: `КурсоваяЧерновик.tex`. Иллюстрации находятся в `figures/`.

Сборка из PowerShell (два прохода нужны для оглавления и ссылок):

```powershell
Push-Location docs/coursework
try {
    xelatex -interaction=nonstopmode -output-directory ..\..\build\coursework 'КурсоваяЧерновик.tex'
    xelatex -interaction=nonstopmode -output-directory ..\..\build\coursework 'КурсоваяЧерновик.tex'
} finally {
    Pop-Location
}
```

Нужен XeLaTeX, шрифт Times New Roman,
поддержка русского языка и класс `extarticle`. Сборочные файлы не лежат рядом
с исходником и исключены из Git.
