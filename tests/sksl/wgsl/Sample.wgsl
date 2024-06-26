diagnostic(off, derivative_uniformity);
diagnostic(off, chromium.unreachable_code);
struct FSOut {
  @location(0) sk_FragColor: vec4<f32>,
};
@group(1) @binding(3) var tex_Sampler: sampler;
@group(1) @binding(2) var tex_Texture: texture_2d<f32>;
fn _skslMain(_stageOut: ptr<function, FSOut>) {
  {
    let a: vec4<f32> = textureSample(tex_Texture, tex_Sampler, vec2<f32>(1.0));
    let _skTemp0 = vec3<f32>(1.0);
    let b: vec4<f32> = textureSample(tex_Texture, tex_Sampler, _skTemp0.xy / _skTemp0.z);
    let _skTemp1 = vec3<f32>(1.0);
    let c: vec4<f32> = textureSampleBias(tex_Texture, tex_Sampler, _skTemp1.xy / _skTemp1.z, -0.75 + 0.0);
    (*_stageOut).sk_FragColor = (a * b) * c;
  }
}
@fragment fn main() -> FSOut {
  var _stageOut: FSOut;
  _skslMain(&_stageOut);
  return _stageOut;
}
