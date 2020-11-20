#version 450
#pragma shader_stage(fragment)
#pragma optimize(on)

layout(location = 0) in vec2 i_UV;
layout(location = 1) in float i_Sign;
layout(location = 2) in float i_Alpha;

layout(location = 0) out vec4 o_Color;

// TODO: Have this be either per-vertex or per-invocation
#define color vec3(1., 1., 1.)

vec4 ifGt(float a, float b, vec4 ifTrue, vec4 ifFalse) {
    float gt = step(a, b);

    return ifFalse * gt + ifTrue * (1 - gt);
}

void main() {
    float distance = i_UV.x * i_UV.x - i_UV.y;
    float discriminant = i_Sign * distance;

    // This ensures that if `i_Sign == 0` we always fill with colour (so we can use the same shader for everything)
    o_Color = ifGt(discriminant, 0, vec4(color, -2.), vec4(color, 1.));
}
