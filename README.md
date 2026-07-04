# Шлиф-Web

Unified web service for **automatic ore classification from reflected-light optical microscopy**
(полированные шлифы / аншлифы) with an **inline expert correction tool** — one continuous flow:

> upload шлиф → automatic result (сорт + фазы + тальк) → **«Доработать»** opens a multi-layer
> mask editor on the same image → edits recompute the verdict and are saved.

Replaces the two Streamlit MVP apps (processing + annotation) with a **granian FastAPI** backend +
**Next.js** frontend behind **Traefik**, in one `docker-compose`.

## Status

Pre-implementation. Design spec:
[`docs/superpowers/specs/2026-07-04-shlif-web-unified-service-design.md`](docs/superpowers/specs/2026-07-04-shlif-web-unified-service-design.md).

Spun out of the hackathon repo `hakaton_nornikel`; the runtime `shlif` pipeline is vendored here,
training scripts stay in the origin repo.

## Stack

- **Backend:** FastAPI served by granian (ASGI), SQLite job store, filesystem for uploads/masks,
  GPU auto-detected with a classical/CPU fallback.
- **Frontend:** Next.js App Router, TanStack Query, HTML-canvas mask editor, «Шлиф» design system.
- **Infra:** Traefik v3 routing `/api → api`, everything else `→ web`.

_No authentication (descoped)._ Runs locally on CPU today; lights up the GPU path unchanged on the
organizer L4 VM when it returns.
