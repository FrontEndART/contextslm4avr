# Patch API

A simple, Docker-based API that serves data from `patches.json`
(VUL4J id → list of patches).

## How it works

In `patches.json`, every id has exactly 10 patches. The API takes an `id` and a
`count` (number) parameter, and returns that many patches from the list:

- if `count <= 10`: it returns the first `count` patches from the list,
- if `count > 10`: it returns all 10 patches, then pads the response to the
  requested length by duplicating the last element (e.g. for `count=13`, the
  10th patch appears 3 extra times at the end, giving 13 elements in total).

## Running with Docker Compose (recommended)

The container joins the external `avr-client` network so that another
(worker) container can reach it at `http://api:8000`. This network needs to
be created **once, in advance**:

```bash
docker network create avr-client
```

Then start the service:

```bash
docker compose up --build
```

If the worker runs as a separate compose project, it also needs to reference
`avr-client` with `external: true` there, and from it this API is reachable
at `http://api:8000` (thanks to `container_name: api`).

## Running with plain Docker

```bash
docker build -t patch-api .
docker run -p 8000:8000 patch-api
```

The API is then available at `http://localhost:8000`.

## Endpoints

### `GET /patches?id=<ID>&count=<N>`

Example:

```bash
curl "http://localhost:8000/patches?id=VUL4J-1&count=5"
curl "http://localhost:8000/patches?id=VUL4J-1&count=13"
```

Response: a JSON array with `count` patch (diff) strings.

If the `id` doesn't exist, it returns `404`.

### `GET /ids`

Returns the list of all available ids.

### `GET /health`

Simple health check, shows how many ids are loaded.

### Interactive documentation

While running, the Swagger UI is available at `http://localhost:8000/docs`.

## File structure

```
patch-api/
├── app/
│   ├── main.py         # the FastAPI application
│   └── data/
│       └── patches.json   # the uploaded data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

If the data file is updated, just replace the contents of
`app/data/patches.json` and rebuild the image
(`docker compose up --build`).
