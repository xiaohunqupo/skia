# Copyright 2019 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# Recipe which runs DM with trace flag on lottie files and then parses the
# trace output into output JSON files to ingest to perf.skia.org.
# Design doc: go/skottie-tracing


import calendar
import json
import re
import string

PYTHON_VERSION_COMPATIBILITY = "PY3"

DEPS = [
  'flavor',
  'infra',
  'recipe_engine/context',
  'recipe_engine/file',
  'recipe_engine/json',
  'recipe_engine/path',
  'recipe_engine/step',
  'recipe_engine/time',
  'recipe_engine/properties',
  'recipe_engine/raw_io',
  'run',
  'vars',
]

SEEK_TRACE_NAME = 'skottie::Animation::seek'
RENDER_TRACE_NAME = 'skottie::Animation::render'
EXPECTED_DM_FRAMES = 25


def perf_steps(api):
  """Run DM on lottie files with tracing turned on and then parse the output."""
  api.flavor.create_clean_device_dir(
        api.flavor.device_dirs.dm_dir)
  lotties_host = api.path.start_dir.joinpath('lotties_with_assets')
  lotties_device = api.path.start_dir.joinpath('lotties_with_assets')
  if 'Android' in api.vars.builder_cfg.get('extra_config'):
    # Due to http://b/72366966 and the fact that CIPD symlinks files in by default, we ran into
    # a strange "Function not implemented" error when trying to copy folders that contained
    # symlinked files. It is not easy to change the CIPD "InstallMode" from symlink to copy, so
    # we use shutil (file.copytree) to make a local copy of the files on the host, which removes
    # the symlinks and adb push --sync works as expected.
    lotties_device = api.flavor.device_path_join(api.flavor.device_dirs.tmp_dir, 'lotties_with_assets')
    api.flavor.create_clean_device_dir(lotties_device)

    # Make a temp directory and then copy to a *non-existing* subfolder (otherwise copytree crashes).
    lotties_no_symlinks = api.path.mkdtemp('lwa').joinpath('nosymlinks')
    api.file.copytree('Copying files on host to remove symlinks', lotties_host, lotties_no_symlinks)
    lotties_host = lotties_no_symlinks
    api.flavor.copy_directory_contents_to_device(lotties_host, lotties_device)

  # We expect this to be a bunch of folders that contain a data.json and optionally
  # an images/ subfolder with image assets required.
  lottie_files = api.file.listdir(
      'list lottie files', lotties_host,
      test_data=['skottie_asset_000', 'skottie_asset_001', 'skottie_asset_002'])
  perf_results = {}
  # Run DM on each lottie file and parse the trace files.
  for idx, lottie_file in enumerate(lottie_files):
    lottie_name = api.path.basename(lottie_file)
    lottie_folder = api.flavor.device_path_join(lotties_device, lottie_name)

    trace_output_path = api.flavor.device_path_join(
        api.flavor.device_dirs.dm_dir, '%s.json' % (idx + 1))
    # See go/skottie-tracing for how these flags were selected.
    dm_args = [
      'dm',
      '--resourcePath', api.flavor.device_dirs.resource_dir,
      '--lotties', lottie_folder,
      '--src', 'lottie',
      '--nonativeFonts',
      '--verbose',
      '--traceMatch', 'skottie',  # recipe can OOM without this.
      '--trace', trace_output_path,
      '--match', get_trace_match(
          'data.json', 'Android' in api.properties['buildername']),
    ]
    if api.vars.builder_cfg.get('cpu_or_gpu') == 'GPU':
      dm_args.extend(['--config', 'gles', '--nocpu'])
    elif api.vars.builder_cfg.get('cpu_or_gpu') == 'CPU':
      dm_args.extend(['--config', '8888', '--nogpu'])
    api.run(api.flavor.step, 'dm', cmd=dm_args, abort_on_failure=False)

    trace_test_data = api.properties.get('trace_test_data', '{}')
    trace_file_content = api.flavor.read_file_on_device(trace_output_path)
    if not trace_file_content and trace_test_data:
      trace_file_content = trace_test_data

    key = 'gles'
    if api.vars.builder_cfg.get('cpu_or_gpu') == 'CPU':
        key = '8888'
    perf_results[lottie_name] = {
        key: parse_trace(trace_file_content, lottie_name, api),
    }
    api.flavor.remove_file_on_device(trace_output_path)

  # Construct contents of the output JSON.
  perf_json = {
      'gitHash': api.properties['revision'],
      'swarming_bot_id': api.vars.swarming_bot_id,
      'swarming_task_id': api.vars.swarming_task_id,
      'renderer': 'skottie',
      'key': {
        'bench_type': 'tracing',
        'source_type': 'skottie',
      },
      'results': perf_results,
  }
  if api.vars.is_trybot:
    perf_json['issue'] = api.vars.issue
    perf_json['patchset'] = api.vars.patchset
    perf_json['patch_storage'] = api.vars.patch_storage
  # Add tokens from the builder name to the key.
  reg = re.compile('Perf-(?P<os>[A-Za-z0-9_]+)-'
                   '(?P<compiler>[A-Za-z0-9_]+)-'
                   '(?P<model>[A-Za-z0-9_]+)-'
                   '(?P<cpu_or_gpu>[A-Z]+)-'
                   '(?P<cpu_or_gpu_value>[A-Za-z0-9_]+)-'
                   '(?P<arch>[A-Za-z0-9_]+)-'
                   '(?P<configuration>[A-Za-z0-9_]+)-'
                   'All(-(?P<extra_config>[A-Za-z0-9_]+)|)')
  m = reg.match(api.properties['buildername'])
  keys = ['os', 'compiler', 'model', 'cpu_or_gpu', 'cpu_or_gpu_value', 'arch',
          'configuration', 'extra_config']
  for k in keys:
    perf_json['key'][k] = m.group(k)

  # Create the output JSON file in perf_data_dir for the Upload task to upload.
  api.file.ensure_directory(
      'makedirs perf_dir',
      api.flavor.host_dirs.perf_data_dir)
  now = api.time.utcnow()
  ts = int(calendar.timegm(now.utctimetuple()))
  json_path = api.flavor.host_dirs.perf_data_dir.joinpath(
      'perf_%s_%d.json' % (api.properties['revision'], ts))
  json_contents = json.dumps(
      perf_json, indent=4, sort_keys=True, separators=(',', ': '))
  api.file.write_text('write output JSON', json_path, json_contents)


def get_trace_match(lottie_filename, is_android):
  """Returns the DM regex to match the specified lottie file name."""
  trace_match = '^%s$' % lottie_filename
  if is_android and ' ' not in trace_match:
    # Punctuation characters confuse DM when shelled out over adb, so escape
    # them. Do not need to do this when there is a space in the match because
    # subprocess.list2cmdline automatically adds quotes in that case.
    for sp_char in string.punctuation:
      if sp_char == '\\':
        # No need to escape the escape char.
        continue
      trace_match = trace_match.replace(sp_char, '\%s' % sp_char)
  return trace_match


def parse_trace(trace_json, lottie_filename, api):
  """parse_trace parses the specified trace JSON.

  Parses the trace JSON and calculates the time of a single frame. Frame time is
  considered the same as seek time + render time.
  Note: The first seek is ignored because it is a constructor call.

  A dictionary is returned that has the following structure:
  {
    'frame_max_us': 100,
    'frame_min_us': 90,
    'frame_avg_us': 95,
  }
  """
  script = api.infra.resource('parse_skottie_trace.py')
  step_result = api.run(
      api.step,
      'parse %s trace' % lottie_filename,
      cmd=['python3', script, trace_json, lottie_filename, api.json.output(),
           SEEK_TRACE_NAME, RENDER_TRACE_NAME, EXPECTED_DM_FRAMES])

  # Sanitize float outputs to 2 precision points.
  output = dict(step_result.json.output)
  output['frame_max_us'] = float("%.2f" % output['frame_max_us'])
  output['frame_min_us'] = float("%.2f" % output['frame_min_us'])
  output['frame_avg_us'] = float("%.2f" % output['frame_avg_us'])
  return output


def RunSteps(api):
  api.vars.setup()
  api.file.ensure_directory('makedirs tmp_dir', api.vars.tmp_dir)
  api.flavor.setup('dm')

  with api.context():
    try:
      api.flavor.install(resources=True, lotties=True)
      perf_steps(api)
    finally:
      api.flavor.cleanup_steps()
    api.run.check_failure()


def GenTests(api):
  trace_output = """
[{"ph":"X","name":"void skottie::Animation::seek(SkScalar)","ts":452,"dur":2.57,"tid":1,"pid":0},{"ph":"X","name":"void SkCanvas::drawPaint(const SkPaint &)","ts":473,"dur":2.67e+03,"tid":1,"pid":0},{"ph":"X","name":"void skottie::Animation::seek(SkScalar)","ts":3.15e+03,"dur":2.25,"tid":1,"pid":0},{"ph":"X","name":"void skottie::Animation::render(SkCanvas *, const SkRect *, RenderFlags) const","ts":3.15e+03,"dur":216,"tid":1,"pid":0},{"ph":"X","name":"void SkCanvas::drawPath(const SkPath &, const SkPaint &)","ts":3.35e+03,"dur":15.1,"tid":1,"pid":0},{"ph":"X","name":"void skottie::Animation::seek(SkScalar)","ts":3.37e+03,"dur":1.17,"tid":1,"pid":0},{"ph":"X","name":"void skottie::Animation::render(SkCanvas *, const SkRect *, RenderFlags) const","ts":3.37e+03,"dur":140,"tid":1,"pid":0}]
"""
  dm_json_test_data = """
{
  "gitHash": "bac53f089dbc473862bc5a2e328ba7600e0ed9c4",
  "swarming_bot_id": "skia-rpi-094",
  "swarming_task_id": "438f11c0e19eab11",
  "key": {
    "arch": "arm",
    "compiler": "Clang",
    "cpu_or_gpu": "GPU",
    "cpu_or_gpu_value": "Mali400MP2",
    "extra_config": "Android",
    "model": "AndroidOne",
    "os": "Android"
   },
   "results": {
   }
}
"""
  parse_trace_json = {
      'frame_avg_us': 179.71,
      'frame_min_us': 141.17,
      'frame_max_us': 218.25
  }
  android_buildername = ('Perf-Android-Clang-AndroidOne-GPU-Mali400MP2-arm-'
                         'Release-All-Android_SkottieTracing')
  gpu_buildername = ('Perf-Debian10-Clang-NUC7i5BNK-GPU-IntelIris640-x86_64-'
                     'Release-All-SkottieTracing')
  cpu_buildername = ('Perf-Debian10-Clang-GCE-CPU-AVX2-x86_64-Release-All-'
                     'SkottieTracing')
  yield (
      api.test(android_buildername) +
      api.properties(buildername=android_buildername,
                     repository='https://skia.googlesource.com/skia.git',
                     revision='abc123',
                     task_id='abc123',
                     trace_test_data=trace_output,
                     dm_json_test_data=dm_json_test_data,
                     path_config='kitchen',
                     swarm_out_dir='[SWARM_OUT_DIR]') +
      api.step_data('parse skottie_asset_000 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_001 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_002 trace',
                    api.json.output(parse_trace_json))
  )
  yield (
      api.test(gpu_buildername) +
      api.properties(buildername=gpu_buildername,
                     repository='https://skia.googlesource.com/skia.git',
                     revision='abc123',
                     task_id='abc123',
                     trace_test_data=trace_output,
                     dm_json_test_data=dm_json_test_data,
                     path_config='kitchen',
                     swarm_out_dir='[SWARM_OUT_DIR]') +
      api.step_data('parse skottie_asset_000 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_001 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_002 trace',
                    api.json.output(parse_trace_json))
  )
  yield (
      api.test(cpu_buildername) +
      api.properties(buildername=cpu_buildername,
                     repository='https://skia.googlesource.com/skia.git',
                     revision='abc123',
                     task_id='abc123',
                     trace_test_data=trace_output,
                     dm_json_test_data=dm_json_test_data,
                     path_config='kitchen',
                     swarm_out_dir='[SWARM_OUT_DIR]') +
      api.step_data('parse skottie_asset_000 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_001 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_002 trace',
                    api.json.output(parse_trace_json))
  )
  yield (
      api.test('skottietracing_parse_trace_error') +
      api.properties(buildername=android_buildername,
                     repository='https://skia.googlesource.com/skia.git',
                     revision='abc123',
                     task_id='abc123',
                     trace_test_data=trace_output,
                     dm_json_test_data=dm_json_test_data,
                     path_config='kitchen',
                     swarm_out_dir='[SWARM_OUT_DIR]') +
      api.step_data('parse skottie_asset_000 trace',
                    api.json.output(parse_trace_json), retcode=1)
  )
  yield (
      api.test('skottietracing_trybot') +
      api.properties(buildername=android_buildername,
                     repository='https://skia.googlesource.com/skia.git',
                     revision='abc123',
                     task_id='abc123',
                     trace_test_data=trace_output,
                     dm_json_test_data=dm_json_test_data,
                     path_config='kitchen',
                     swarm_out_dir='[SWARM_OUT_DIR]',
                     patch_ref='89/456789/12',
                     patch_repo='https://skia.googlesource.com/skia.git',
                     patch_storage='gerrit',
                     patch_set=7,
                     patch_issue=1234,
                     gerrit_project='skia',
                     gerrit_url='https://skia-review.googlesource.com/') +
      api.step_data('parse skottie_asset_000 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_001 trace',
                    api.json.output(parse_trace_json)) +
      api.step_data('parse skottie_asset_002 trace',
                    api.json.output(parse_trace_json))
  )
