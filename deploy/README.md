# Deploying INSTRUMENTAL

The page is static and lives on Cloudflare Pages with the rest of
philippbogdan.com. The matching runs on the Mac mini, reached through a
Cloudflare tunnel, so no port is opened and the home address is never exposed.

```
browser ──> philippbogdan.com/instrumental        (Pages, always up)
              │
              ├─ demos ─────────────────────────> static audio, always play
              │
              └─ /api, /ws ──> api.philippbogdan.com
                                 │  Cloudflare tunnel
                                 └──> 127.0.0.1:8801 on the mini
                                        Demucs (MPS)  ->  CMA-ES (MPS)

  mini off  ->  the fetch fails  ->  body.backend-offline  ->  the page says so
                and hides the controls that cannot work. The demos stay.
```

## One-time: the tunnel

This needs a browser sign-in to Cloudflare, so it is the one step that has to
be run by hand.

```sh
cloudflared tunnel login                      # pick philippbogdan.com
cloudflared tunnel create instrumental
cloudflared tunnel route dns instrumental api.philippbogdan.com
mkdir -p ~/.cloudflared && cp deploy/cloudflared-config.yml ~/.cloudflared/config.yml
# put the tunnel's UUID into that file, then:
sudo cloudflared service install                # or run it under launchd, below
```

## The backend service

```sh
cp deploy/com.philippbogdan.instrumental.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.philippbogdan.instrumental.plist
curl -s localhost:8801/api/health               # {"status":"ok"}
```

Idle it holds about 400 MB, which is torch and the filterbanks. A match adds
roughly half a gigabyte for the optimiser and a short spike of about two for
separation, one job at a time. Measured on an M4 with 16 GB:

| stage | wall | notes |
| --- | --- | --- |
| Demucs, 30 s clip | ~7 s | `htdemucs` on MPS, `--segment 7` |
| CMA-ES, 10,000 evals | ~11 s | batched on MPS, one process |

## The page

```sh
./publish.sh                                    # writes ~/personal-website/public/instrumental
cd ~/personal-website && npm run deploy
```

`publish.sh` writes `config.js` with the API host. Override it for a local
backend:

```sh
INSTRUMENTAL_API= ./publish.sh                  # same origin, for development
```

## Guards

The service is on a home connection, so the app throttles rather than trusting
the internet: twelve heavy requests an hour per address, a queue capped at six,
and one Demucs and one CMA-ES at a time. Both numbers are environment variables
in the plist.
