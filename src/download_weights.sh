#/bin/bash

# Download weights from gdrive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz" -O checkpoints/segformer.b5.640x640.ade.160k.pth && rm -rf /tmp/cookies.txt