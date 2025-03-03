precision highp float;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D texture;
uniform float time;

attribute vec3 position;
attribute vec2 uv;
attribute vec3 translate;
attribute vec2 prop;

varying vec2 vUv;
varying vec4 vColor;
varying float vScale;


void main() {
    vUv = uv;

    vec4 mvPosition = modelViewMatrix * vec4(translate, 1.0);
    mvPosition.xyz += position;

    float textureIdx = prop.x;
    float textureDim = 288.0;

    float i = mod(textureIdx, textureDim) / (textureDim - 1.0);
    float j = floor(textureIdx / textureDim) / (textureDim - 1.0);

    vec4 textureData = texture2D(texture, vec2(i, j));
    float value = clamp(decodeRGBAFloat(textureData), 0.0, 1.0);

    float alpha = 0.75;
    float scale = clamp(value, 0.1, 0.5);

    vColor = vec4(viridis(value).xyz, alpha);
    vScale = scale;

    gl_Position = projectionMatrix * mvPosition;
}
