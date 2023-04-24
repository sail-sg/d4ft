import glob
import gzip
import http.server
import json
import os
import socketserver
from typing import Optional

from absl import app, flags, logging
from jax._src import traceback_util

traceback_util.register_exclusion(__file__)

from jax._src.lib import xla_client

_profiler_server: Optional[xla_client.profiler.ProfilerServer] = None

FLAGS = flags.FLAGS
flags.DEFINE_string("gz_dir", "jax-trace", "")
flags.DEFINE_integer("offset", -1, "default to select latest run")


def _write_perfetto_trace_file(log_dir):
  # Navigate to folder with the latest trace dump to find `trace.json.jz`
  curr_path = os.path.abspath(log_dir)
  root_trace_folder = os.path.join(curr_path, "plugins", "profile")
  trace_folders = [
    os.path.join(root_trace_folder, trace_folder)
    for trace_folder in os.listdir(root_trace_folder)
  ]
  sorted_folder = sorted(trace_folders, key=os.path.getmtime)
  latest_folder = sorted_folder[FLAGS.offset]
  trace_jsons = glob.glob(os.path.join(latest_folder, "*.trace.json.gz"))
  if len(trace_jsons) != 1:
    raise ValueError(f"Invalid trace folder: {latest_folder}")
  trace_json, = trace_jsons

  logging.info(f"dir: {latest_folder} {trace_json}")
  logging.info("Loading trace.json.gz and removing its metadata...")
  # Perfetto doesn't like the `metadata` field in `trace.json` so we remove
  # it.
  # TODO(sharadmv): speed this up by updating the generated `trace.json`
  # to not include metadata if possible.
  with gzip.open(trace_json, "rb") as fp:
    trace = json.load(fp)
    del trace["metadata"]
  filename = "perfetto_trace.json.gz"
  perfetto_trace = os.path.join(latest_folder, filename)
  logging.info("Writing perfetto_trace.json.gz...")
  with gzip.open(perfetto_trace, "w") as fp:
    fp.write(json.dumps(trace).encode("utf-8"))
  return perfetto_trace


def _host_perfetto_trace_file(log_dir):
  # ui.perfetto.dev looks for files hosted on `127.0.0.1:9001`. We set up a
  # TCP server that is hosting the `perfetto_trace.json.gz` file.
  port = 9001
  abs_filename = _write_perfetto_trace_file(log_dir)
  orig_directory = os.path.abspath(os.getcwd())
  directory, filename = os.path.split(abs_filename)
  try:
    os.chdir(directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', port), _PerfettoServer) as httpd:
      url = (
        f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{filename}"
      )
      print(f"Open URL in browser: {url}")

      # Once ui.perfetto.dev acquires trace.json from this server we can close
      # it down.
      while httpd.__dict__.get('last_request') != '/' + filename:
        httpd.handle_request()
  finally:
    os.chdir(orig_directory)


class _PerfettoServer(http.server.SimpleHTTPRequestHandler):
  """Handles requests from `ui.perfetto.dev` for the `trace.json`"""

  def end_headers(self):
    self.send_header('Access-Control-Allow-Origin', '*')
    return super().end_headers()

  def do_GET(self):
    self.server.last_request = self.path
    return super().do_GET()

  def do_POST(self):
    self.send_error(404, "File not found")


def main(_):
  _host_perfetto_trace_file(FLAGS.gz_dir)


if __name__ == '__main__':
  app.run(main)
