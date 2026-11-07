# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' PYTHONWARNINGS=ignore && source .venv/bin/activate` Enter
Sleep 3s
Type `python -m privacy_attacks_lab serve --port 18415 >/dev/null 2>&1 &` Enter
Sleep 8s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18415/ready | jq -c` Enter
Sleep 2s
Type `curl -s localhost:18415/epsilon -d '{"q": 0.004, "sigma": 1.1, "steps": 7500}' | jq -c` Enter
Sleep 4s
Type `curl -s localhost:18415/exposure -d '{"rank": 3, "space": 100000}' | jq -c` Enter
Sleep 2s
Type `curl -s localhost:18415/metrics | grep pal_requests_total` Enter
Sleep 5s
Hide
Type `kill %1` Enter
Sleep 1s
