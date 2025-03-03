precision highp float;

varying vec2 vUv;
varying vec4 vColor;
varying float vScale;

vec4 sdfCircle(vec2 uv, float r) {
    float x = uv.x - 0.5;
    float y = uv.y - 0.5;
    float d = length(vec2(x, y)) - r;
    vec4 col = vec4(1.0, 1.0, 1.0, 0.);
    col += vec4(0., 0., 0., step(0., -d));
    return col;
}

void main() {
    vec2 st = vUv;
    vec4 res = sdfCircle(st, vScale) * vColor;
    if (res.w < 0.1) discard;
    gl_FragColor = res;
}
