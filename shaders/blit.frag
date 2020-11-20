#version 450
#pragma shader_stage(fragment)
#pragma optimize(on)

layout(location = 0) in vec2 i_UV;

layout(location = 0) out vec4 o_Color;

layout(set = 0, binding = 0) uniform sampler u_Color;
layout(set = 0, binding = 1) uniform texture2D u_Diffuse;

void main() {
    vec4 color = texture(
        sampler2D(u_Diffuse, u_Color),
        i_UV
    );

    o_Color = vec4(color.rgb, max(color.a, 0));
}
