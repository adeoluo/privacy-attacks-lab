# Records docs/demo/accounting.gif from a real run: `vhs docs/demo/accounting.tape` from the repo root.
Output docs/demo/accounting.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 620
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' PYTHONWARNINGS=ignore && source .venv/bin/activate` Enter
Sleep 3s
Type `clear` Enter
Sleep 1s
Show
Type `python -m privacy_attacks_lab epsilon --q 0.004 --sigma 1.1 --epochs 30 --accountant rdp` Enter
Sleep 10s
Type `python -m privacy_attacks_lab calibrate --q 0.004 --epochs 30 --target-epsilon 1` Enter
Sleep 10s
Type `python -m privacy_attacks_lab max-tpr --epsilon 1` Enter
Sleep 12s
