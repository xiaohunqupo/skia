diagnostic(off, derivative_uniformity);
diagnostic(off, chromium.unreachable_code);
struct VSIn {
  @builtin(instance_index) sk_InstanceID: u32,
};
struct VSOut {
  @location(1) @interpolate(flat, either) id: i32,
  @builtin(position) sk_Position: vec4<f32>,
};
fn fn_i(_stageIn: VSIn) -> i32 {
  {
    return i32(_stageIn.sk_InstanceID);
  }
}
fn _skslMain(_stageIn: VSIn, _stageOut: ptr<function, VSOut>) {
  {
    let _skTemp0 = fn_i(_stageIn);
    (*_stageOut).id = _skTemp0;
  }
}
@vertex fn main(_stageIn: VSIn) -> VSOut {
  var _stageOut: VSOut;
  _skslMain(_stageIn, &_stageOut);
  return _stageOut;
}
