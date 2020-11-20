#version 450
#pragma shader_stage(vertex)
#pragma optimize(on)

layout(location = 0) in vec2 i_Position;
layout(location = 1) in vec2 i_UV;

layout(location = 0) out vec2 o_UV;

void main() {
    gl_Position = vec4(i_Position, 0., 1.);
    o_UV = i_UV;
}
