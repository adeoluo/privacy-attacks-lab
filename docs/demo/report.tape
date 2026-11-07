# Records docs/demo/report.gif from a real run: `vhs docs/demo/report.tape` from the repo root.
Output docs/demo/report.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 600
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' PYTHONWARNINGS=ignore && source .venv/bin/activate` Enter
Sleep 3s
Type `clear` Enter
Sleep 1s
Show
Type `python -m privacy_attacks_lab report benchmarks/results/mia_cifar10.json --sort tpr@0.001` Enter
Sleep 14s
Type `python -m privacy_attacks_lab report benchmarks/results/mia_purchase100.json --sort tpr@0.001` Enter
Sleep 16s
