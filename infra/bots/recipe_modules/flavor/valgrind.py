# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


from . import default


"""Valgrind flavor, used for running code through Valgrind."""


class ValgrindFlavor(default.DefaultFlavor):
  def __init__(self, m, app_name):
    super(ValgrindFlavor, self).__init__(m, app_name)
    self._suppressions_file = self.m.path.start_dir.joinpath(
        'skia', 'tools', 'valgrind.supp')
    self._valgrind_cipd_dir = self.m.vars.workdir.joinpath('valgrind')
    self._valgrind_fake_dir = self._valgrind_cipd_dir
    self._valgrind = self._valgrind_fake_dir.joinpath('bin', 'valgrind')
    self._lib_dir = self._valgrind_fake_dir.joinpath('lib', 'valgrind')

  def step(self, name, cmd, **kwargs):
    new_cmd = [self._valgrind, '--gen-suppressions=all', '--leak-check=full',
               '--track-origins=yes', '--error-exitcode=1', '--num-callers=40',
               '--vex-guest-max-insns=25',
               '--suppressions=%s' % self._suppressions_file]
    path_to_app = self.host_dirs.bin_dir.joinpath(cmd[0])
    new_cmd.append(path_to_app)
    new_cmd.extend(cmd[1:])
    with self.m.env({'VALGRIND_LIB': self._lib_dir}):
      return self.m.run(self.m.step, name, cmd=new_cmd, **kwargs)
