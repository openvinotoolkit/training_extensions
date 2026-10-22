# Geti MCP server

A [Model Context Protocol](https://modelcontextprotocol.io) server that lets an AI assistant
inspect an existing Geti™ project, start training, monitor jobs, and read model results.

It is an **application integration**: every operation goes through the Geti REST API. It never
imports backend services, touches the database, or reads Geti's data directory, and it never
launches, restarts, migrates, or resets Geti. Geti must already be running.

- **Transport:** stdio only. The MCP host launches the process.
- **Default posture:** read-only. Training, job cancellation, and image disclosure each require a
  separate, explicit opt-in.
- **Declared compatibility:** Geti API `3.2.x` (see [Compatibility](#compatibility)).

## How it fits together

The server is a _client_ of Geti, launched by your AI assistant and running on the same machine
as the assistant:

```
[MCP host: Claude Desktop, VS Code, …]  --stdio-->  [geti-mcp]  --HTTPS-->  [Geti REST API]
```

It is a separate process from the Geti desktop app and is not bundled with it. Geti must
already be running; the server will not start it.

## Quick start: Windows desktop app

The desktop app serves the same REST API as any other Geti install, on `https://localhost:7860`.
The Geti connection stays on your machine — no tunnel or network exposure is needed for the local desktop app.

**1. Start Geti.** If the app is not running, every tool reports `backend_unavailable`.

**2. Install the server.** It needs its own Python 3.11–3.14 interpreter. A dedicated virtual
environment keeps the executable at a predictable path, which matters because MCP hosts do not
read your `PATH`:

```powershell
py -m venv $env:USERPROFILE\geti-mcp-venv
& $env:USERPROFILE\geti-mcp-venv\Scripts\pip install "C:\path\to\training_extensions\integrations\geti-mcp"
& $env:USERPROFILE\geti-mcp-venv\Scripts\geti-mcp.exe --help
```

The executable is then at `C:\Users\<you>\geti-mcp-venv\Scripts\geti-mcp.exe`.

**3. Find the certificate.** Geti generates a self-signed certificate on first run and keeps it
in its per-user data directory. The packaged Windows backend pins that directory to
`%LOCALAPPDATA%\Intel\Geti`, so the certificate is at:

```
%LOCALAPPDATA%\Intel\Geti\certs\localhost.pem
```

If it is not there, the app has not been started yet, or `DATA_DIR` was overridden. Search for
it:

```powershell
Get-ChildItem $env:LOCALAPPDATA -Recurse -Filter localhost.pem -ErrorAction SilentlyContinue |
  Select-Object FullName
```

**4. Point your assistant at it.** For Claude Desktop, edit
`%APPDATA%\Claude\claude_desktop_config.json`. Use absolute paths with escaped backslashes —
environment variables are **not** expanded in this file:

```json
{
  "mcpServers": {
    "geti": {
      "command": "C:\\Users\\you\\geti-mcp-venv\\Scripts\\geti-mcp.exe",
      "args": [
        "--base-url",
        "https://localhost:7860",
        "--all-projects",
        "--ca-bundle",
        "C:\\Users\\you\\AppData\\Local\\Intel\\Geti\\certs\\localhost.pem"
      ]
    }
  }
}
```

**5. Verify.** Restart the host, then ask the assistant to run `get_connection_info`. A working
setup reports `reachable: true`, an `api_version` of `3.2.x`, and `compatible: true`. Follow up
with _"list my Geti projects"_.

**6. Tighten the scope.** `--all-projects` is the easiest way to bootstrap, because
`list_projects` is how you discover your project IDs in the first place. Once you have them,
replace it with `--projects <id>,<id>` and add permission flags only as you need them.

## Other setups

### macOS and Linux, from a checkout

```bash
cd integrations/geti-mcp
just venv
.venv/bin/geti-mcp --help
```

That creates `.venv/` containing the `geti-mcp` executable; use its absolute path in the host
configuration. The package depends only on `mcp`, `httpx`, `pydantic`, and `pillow` — it does
not pull in the Geti backend or the `getitune` training library.

### A Geti instance on another machine

Two things make a direct remote connection impractical, and one fix solves both.

Geti's self-signed certificate is issued for `localhost` and `127.0.0.1` only, and this server
always verifies hostnames — there is no flag to disable that, so connecting to
`https://some-host:7860` fails verification regardless of `--ca-bundle`. Separately, the Geti
API has **no authentication**: anyone who can reach port 7860 controls the instance.

Forward the port over SSH instead:

```bash
ssh -N -L 7860:localhost:7860 you@your-server
```

Geti now appears at `https://localhost:7860`, the certificate matches, and the port is never
exposed. Copy the certificate across with `scp` rather than fetching it from the connection you
are trying to authenticate:

```bash
scp you@your-server:<data-dir>/certs/localhost.pem ~/geti-ca.pem
```

### Smoke-testing without an MCP host

The handshake and tool list do not depend on Geti being reachable, so this is a quick way to
confirm the executable, flags, and certificate are right:

```bash
.venv/bin/python - <<'PY'
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(
    command=".venv/bin/geti-mcp",
    args=["--base-url", "https://localhost:7860", "--all-projects", "--ca-bundle", "/path/to/localhost.pem"],
)

async def main():
    async with stdio_client(PARAMS) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        print("server:", info.server_info.name, info.server_info.version)
        tools = await session.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))
        print((await session.call_tool("get_connection_info", {})).structured_content)

asyncio.run(main())
PY
```

## Configuration

Every option can be supplied as a command-line flag or as an environment variable prefixed with
`GETI_MCP_` (for example `--base-url` ↔ `GETI_MCP_BASE_URL`). Flags win over environment
variables.

### Required

| Option                           | Description                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `--base-url`                     | Geti base URL, e.g. `https://localhost:7860`. Must be `http://` or `https://`.             |
| `--projects` \| `--all-projects` | Project scope. Mutually exclusive, and one is **required** — there is no implicit default. |

`--projects` takes a comma-separated list of project UUIDs. `--all-projects` authorizes every
project on the instance and should be used deliberately.

### Permissions (all default to off)

| Option                     | Enables                                         |
| -------------------------- | ----------------------------------------------- |
| `--allow-training`         | `start_training`                                |
| `--allow-job-cancellation` | `cancel_job`                                    |
| `--allow-image-access`     | `view_media` — sends image content to the model |

A disabled tool is not registered _and_ its underlying operation is refused, so a client that
invokes a hidden tool name directly is still denied.

### TLS

| Option        | Default      | Description                                                            |
| ------------- | ------------ | ---------------------------------------------------------------------- |
| `--ca-bundle` | system store | PEM certificate or CA bundle to trust in addition to the system store. |

Certificate and hostname verification are **always on**; there is no flag to disable them. A
default Geti install serves a self-signed certificate, so `--ca-bundle` must point at it. The
file is always `certs/localhost.pem` inside Geti's data directory, but that directory differs
per install:

| Install                     | Path                                                               |
| --------------------------- | ------------------------------------------------------------------ |
| Windows desktop app         | `%LOCALAPPDATA%\Intel\Geti\certs\localhost.pem`                    |
| macOS desktop app           | `~/Library/Application Support/com.intel.geti/certs/localhost.pem` |
| Linux desktop app           | `~/.local/share/com.intel.geti/certs/localhost.pem`                |
| Run from source             | `application/backend/data/certs/localhost.pem`                     |
| Docker or custom `DATA_DIR` | `<data-dir>/certs/localhost.pem`                                   |

The packaged Windows backend pins `DATA_DIR` to `%LOCALAPPDATA%\Intel\Geti` regardless of the
bundle identifier, so it does not follow the `com.intel.geti` convention the other platforms use.

The bundle is added to this process's trust context rather than replacing the system trust store, so a Geti behind a properly issued certificate needs no flag at all. An unreadable or malformed bundle fails at
startup with an `invalid_input` error rather than silently falling back to an unverified
connection.

### Proxies

The server honours the standard `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` environment
variables. On a machine configured for a corporate proxy this means a request to a **local**
Geti is sent to the proxy and fails, so the loopback address must be excluded:

```
NO_PROXY=localhost,127.0.0.1
```

An MCP host launches the server as a child process, so the server inherits the host's
environment. A host started from a desktop launcher rather than a shell may not see variables
set in a shell profile; in that case set them in the server's own `env` block:

```json
"env": { "NO_PROXY": "localhost,127.0.0.1" }
```

### Limits

| Option                    | Default   | Description                                                                 |
| ------------------------- | --------- | --------------------------------------------------------------------------- |
| `--connect-timeout`       | `5.0`     | Connection timeout, seconds.                                                |
| `--request-timeout`       | `30.0`    | Per-request timeout, seconds.                                               |
| `--max-response-bytes`    | `4194304` | Hard cap on any response body; larger responses are refused, not truncated. |
| `--max-page-size`         | `50`      | Upper bound on `limit` for paginated tools (Geti itself caps at 100).       |
| `--max-items`             | `200`     | Upper bound on items in any list result.                                    |
| `--max-wait-seconds`      | `20.0`    | Upper bound on a single `wait_for_job` call.                                |
| `--preview-max-dimension` | `768`     | Longest edge of a returned image preview, pixels.                           |
| `--preview-max-bytes`     | `1048576` | Size cap on a returned image preview.                                       |

Out-of-range values are rejected at startup rather than clamped, so a misconfiguration is
visible immediately.

### Logging

`--log-level` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default `INFO`). **All
logs go to stderr.** Stdout carries only MCP protocol messages; anything else there would
corrupt the session.

## MCP host configuration

Always give the **absolute path** to the executable. Hosts launch it directly and do not
inherit your shell environment, so a bare `geti-mcp` usually fails to start.

Claude Desktop uses the key `mcpServers`, in
`%APPDATA%\Claude\claude_desktop_config.json` or
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geti": {
      "command": "/path/to/integrations/geti-mcp/.venv/bin/geti-mcp",
      "args": [
        "--base-url",
        "https://localhost:7860",
        "--projects",
        "7b073838-99d3-42ff-9018-4e901eb047fc",
        "--ca-bundle",
        "/path/to/localhost.pem"
      ]
    }
  }
}
```

VS Code uses the key `servers` and an explicit `type`, in `.vscode/mcp.json`:

```json
{
  "servers": {
    "geti": {
      "type": "stdio",
      "command": "/path/to/integrations/geti-mcp/.venv/bin/geti-mcp",
      "args": [
        "--base-url",
        "https://localhost:7860",
        "--projects",
        "7b073838-99d3-42ff-9018-4e901eb047fc",
        "--ca-bundle",
        "/path/to/localhost.pem"
      ]
    }
  }
}
```

Because permissions are per process, registering two entries is a convenient way to control
blast radius — a read-only `geti` across all projects, and a `geti-train` limited to one
project with the mutating flags enabled:

```json
{
  "mcpServers": {
    "geti-train": {
      "command": "/path/to/geti-mcp",
      "args": [
        "--base-url",
        "https://localhost:7860",
        "--projects",
        "7b073838-99d3-42ff-9018-4e901eb047fc",
        "--allow-training",
        "--allow-job-cancellation"
      ],
      "env": { "GETI_MCP_LOG_LEVEL": "DEBUG" }
    }
  }
}
```

You then choose what the assistant can do by choosing which server to enable.

## Tools

Twelve read-only tools are always registered:

| Tool                         | Returns                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `get_connection_info`        | Reachability, API version and compatibility, enabled permissions, project scope, training devices. |
| `list_projects`              | Authorized projects only.                                                                          |
| `get_project`                | Task type, labels, pipeline, creation time.                                                        |
| `get_dataset_statistics`     | Media and annotation counts, instances per label, readiness warnings.                              |
| `list_media`                 | A bounded page of dataset items (metadata only, no pixels).                                        |
| `list_model_architectures`   | Architectures valid for the project's task, with Geti's own top picks flagged.                     |
| `get_training_configuration` | Effective parameter values with names, defaults, and bounds.                                       |
| `list_jobs`                  | Jobs belonging to authorized projects.                                                             |
| `get_job`                    | One job, with a normalized lifecycle state.                                                        |
| `wait_for_job`               | Polls until the job is terminal or the wait budget expires.                                        |
| `list_models`                | Trained models with variants and a headline metric.                                                |
| `get_model_results`          | Evaluations grouped by variant, plus what is missing or incomparable.                              |

Three more appear only when opted in: `view_media` (`--allow-image-access`), `start_training`
(`--allow-training`), and `cancel_job` (`--allow-job-cancellation`).

### Example prompts

These exercise the surface roughly in the order you would use it:

- _"Is Geti reachable, and what version?"_ — always the first thing to try when debugging.
- _"List my projects, and show me the labels on the detection one."_
- _"Is that dataset ready to train? Any warnings?"_
- _"Which architectures can I use, and what does Geti recommend for speed?"_
- _"Start training with the balanced pick on cpu, then wait for it to finish."_
- _"Compare the evaluation results across the model variants."_

### Job lifecycle

Geti's statuses are normalized so an assistant does not have to special-case backend naming:

| Geti status           | Normalized state | Terminal |
| --------------------- | ---------------- | -------- |
| `PENDING`             | `queued`         | no       |
| `RUNNING`             | `running`        | no       |
| `CANCELLING`          | `cancelling`     | no       |
| `DONE`                | `succeeded`      | yes      |
| `FAILED`              | `failed`         | yes      |
| `CANCELLED`           | `cancelled`      | yes      |
| unrecognized / absent | `unknown`        | no       |

`cancelling` is deliberately **not** terminal: requesting cancellation is not the same as the
job having stopped.

### Errors

Failures are returned as MCP tool errors carrying a JSON body with a stable `error_code`, a
message, and — where an assistant can act on it — `guidance`.

| `error_code`                 | Meaning                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| `permission_denied`          | Project outside scope, or the required opt-in is not enabled.                               |
| `invalid_input`              | Malformed argument, or Geti rejected the request as invalid.                                |
| `not_found`                  | Geti has no such project, job, model, or media item.                                        |
| `ownership_unknown`          | A job's owning project could not be determined, so access was refused.                      |
| `limit_exceeded`             | The response exceeded a configured bound.                                                   |
| `unsupported_media`          | The media item cannot be rendered as a preview.                                             |
| `backend_unavailable`        | Geti could not be reached.                                                                  |
| `backend_error`              | Geti returned an unexpected error.                                                          |
| `incompatible_backend`       | Geti's API version is outside the supported range.                                          |
| `submission_outcome_unknown` | A non-idempotent request failed in flight; **do not resubmit** — list jobs and check first. |

## Security notes

- **Read-only by default.** Nothing mutates state unless the operator opts in.
- **Deny by default on ambiguity.** If a job's owning project cannot be established, access is
  refused rather than granted.
- **Authorization is enforced in the service layer**, not by tool registration, so hidden-tool
  invocation gains nothing.
- **TLS verification cannot be turned off.**
- **Secrets are redacted** from error text and echoed values.
- **Image disclosure is explicit.** `view_media` sends actual image content to the model, so it
  is gated behind its own opt-in and every response carries a disclosure notice.
- **Geti content is data, not instructions.** Project, label, file, job, and model names are
  user-supplied. The server's instructions tell the model never to follow directions found in
  them, but an operator should still treat a Geti instance as untrusted input.

## Compatibility

The server targets Geti API **3.2.x**. It reads `/api/openapi.json` on first use and reports the
result through `get_connection_info` rather than refusing to start, so a version mismatch is
visible without being fatal for read-only work. Training submission is the exception: it is
refused outright against an untested API version, because a silently changed request contract
could start the wrong job.

`tests/assets/geti-openapi-snapshot.json` is a trimmed copy of the contract the adapter was
built against. The contract tests validate endpoints, required parameters, enum values, and
representative payloads against it, so a backend change that breaks this adapter fails a test
rather than failing in production. Refresh it after a backend API change:

```bash
just update-contract
```

To check the adapter against a spec without committing it:

```bash
cd ../../application/backend && just gen-api-spec --output-path /tmp/geti-openapi.json
cd - && GETI_MCP_OPENAPI_SPEC=/tmp/geti-openapi.json just test-integration
```

## Troubleshooting

Logs go to **stderr only**, so read them in your host's server output pane. Raise the level
with `"env": { "GETI_MCP_LOG_LEVEL": "DEBUG" }`.

 | Symptom | Cause |
 | ------- | ----- |
 | Server exits immediately with code 2 | Neither `--projects` nor `--all-projects` was given, or `--base-url` is malformed. Run the command by hand to see the message. |
 | Host reports the server failed to start | `command` is not an absolute path, or points at an interpreter that lacks the package. |
 | `backend_unavailable` mentioning TLS | Wrong or missing `--ca-bundle`, or you are connecting to a remote host by name instead of through a tunnel. |
 | `backend_unavailable` with a connection refusal | Geti is not running, or is on a different port. |
 | `backend_unavailable` against a local Geti on a corporate network | `HTTPS_PROXY` is set and `NO_PROXY` does not list `localhost,127.0.0.1`, so the request goes to the proxy. |
 | `permission_denied` | The project is outside `--projects`, or the matching opt-in flag is off. |
 | `incompatible_backend` on `start_training` | Geti's API version is outside the tested range. Read-only tools still work. |
 | Tool is missing from the host's tool list | Its opt-in flag is not set. Check `get_connection_info`, which reports the enabled permissions. |                             |

## Development

```bash
just lint             # ruff check, ruff format --check, pyrefly
just test-unit        # unit tests
just test-integration # stdio transport + REST contract tests
```

No running Geti instance is required for any test: the unit tests drive a fake backend through
an `httpx` transport, and the integration tests serve that same fake over real HTTP while
launching the actual `geti-mcp` executable as a subprocess.
