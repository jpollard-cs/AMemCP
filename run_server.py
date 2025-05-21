#!/usr/bin/env python3
"""Lightweight launcher for the MCP server

*   Parses a minimal CLI (host, port, OTLP endpoint, etc.).
*   Sets env vars for the underlying `AMemMCPServer`.
*   Configures the OpenTelemetry OTLP HTTP exporter (if `--otel-endpoint` is given).
*   Delegates everything else to `amem_mcp_server_refactored.AMemMCPServer`.

Usage::

    python run_amem_server.py --host 0.0.0.0 --port 8010 \
        --project myproj --data-dir ./data \
        --otel-endpoint http://localhost:4318 --debug-monitor

If you *only* need default values you can just run ``python -m amem_mcp_server_refactored``
which already has a ``__main__`` guard that calls ``server.run()``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from amem.server.mcp_fastmcp_server import AMemCPServer, ServerConfig  # pylint: disable=wrong-import-position

# Ensure local imports work when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOG = logging.getLogger("run_amem_server")


def _parse_cli() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Launch A‑Mem MCP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--project", help="Project / collection prefix")
    parser.add_argument("--data-dir", help="Chroma persistence directory")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--otel-endpoint", help="OTLP HTTP endpoint, e.g. http://localhost:4318")
    parser.add_argument("--service-name", default="amem-mcp-server")
    parser.add_argument("--debug-monitor", action="store_true", help="Enable aiomonitor on 127.0.0.1:50101")
    return parser.parse_args()


@dataclass
class _OTelCfg:
    endpoint: str
    service_name: str


def _configure_otlp(cfg: _OTelCfg) -> None:
    """Wire the global tracer provider to an OTLP HTTP exporter."""
    resource = Resource(attributes={SERVICE_NAME: cfg.service_name})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=f"{cfg.endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    _LOG.info("OTLP exporter initialised → %s", cfg.endpoint)


def _bootstrap_env(args: argparse.Namespace) -> None:
    """Translate selected CLI flags into env vars consumed by ServerConfig."""
    if args.project:
        os.environ["PROJECT_NAME"] = args.project
    if args.data_dir:
        os.environ["PERSIST_DIRECTORY"] = args.data_dir
    if args.debug_monitor:
        os.environ["DEBUG_MONITOR"] = "1"


def main():  # noqa: D401
    args = _parse_cli()

    # logging early so that OTel emits spans later
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.otel_endpoint:
        _configure_otlp(_OTelCfg(endpoint=args.otel_endpoint, service_name=args.service_name))

    _bootstrap_env(args)

    cfg = ServerConfig(host=args.host, port=args.port, debug_monitor=args.debug_monitor)
    server = AMemCPServer(cfg)
    _LOG.info("🚀  AMem MCP server starting at http://%s:%d …", cfg.host, cfg.port)
    try:
        server.run()
    except KeyboardInterrupt:
        _LOG.info("👋  Shutdown via ctl-c")


main()
