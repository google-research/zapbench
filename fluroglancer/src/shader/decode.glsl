const vec4 enc = vec4(1., 255., 65025., 16581375.);
const vec4 dec = 1./enc;

float decodeRGBAFloat(vec4 v) {
    return dot(v, dec);
}
