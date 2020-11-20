trait ShaderModuleSourceExt<'a> {
    fn as_ref<'b: 'a>(&'b self) -> wgpu::ShaderModuleSource<'b>;
}

impl<'a> ShaderModuleSourceExt<'a> for wgpu::ShaderModuleSource<'a> {
    fn as_ref<'b: 'a>(&'b self) -> wgpu::ShaderModuleSource<'b> {
        match self {
            Self::SpirV(data) => Self::SpirV(std::borrow::Cow::Borrowed(&**data)),
            Self::Wgsl(data) => Self::Wgsl(std::borrow::Cow::Borrowed(&**data)),
        }
    }
}

pub struct Pipeline<BindGroup = wgpu::BindGroup> {
    pub bind_group: BindGroup,
    pub pipeline: wgpu::RenderPipeline,
}

#[derive(Default, Clone)]
struct BindId {
    cur: u16,
}

impl BindId {
    fn next<T>(&mut self) -> T
    where
        T: From<u16>,
    {
        let out = self.cur;
        self.cur += 1;
        out.into()
    }
}

const WINDING_MODE: wgpu::FrontFace = wgpu::FrontFace::Ccw;

pub const DEPTH_FORMAT: Option<wgpu::TextureFormat> = None;
pub const SWAPCHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// This is responsible for rendering the curves themselves. The XOR behaviour of overlapping
/// SVGs are handled via the `blit` pass.
pub mod curve {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    use super::BindId;

    pub type Pipeline = super::Pipeline<()>;

    use memoffset::offset_of;
    use std::mem;

    #[derive(Copy, Clone, PartialEq)]
    pub struct Vertex {
        pub pos: [f32; 2],
        pub uv: [f32; 2],
        pub sign: f32,
    }

    unsafe impl bytemuck::Pod for Vertex {}
    unsafe impl bytemuck::Zeroable for Vertex {}

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/curve.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/curve.frag.spv"));
    }

    const COLOR_STATES: &[wgpu::ColorStateDescriptor] = &[wgpu::ColorStateDescriptor {
        format: super::blit::INTERMEDIATE_TEXTURE_FORMAT,
        color_blend: wgpu::BlendDescriptor::REPLACE,
        alpha_blend: wgpu::BlendDescriptor {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        write_mask: wgpu::ColorWrite::ALL,
    }];

    pub fn build(device: &wgpu::Device) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_curve"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let mut ids = BindId::default();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                // We need `::None` so that curves can curve both clockwise and anticlockwise
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: COLOR_STATES,
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, pos) as u64,
                            shader_location: ids.next(),
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, uv) as u64,
                            shader_location: ids.next(),
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float,
                            offset: offset_of!(Vertex, sign) as u64,
                            shader_location: ids.next(),
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Pipeline {
            bind_group: (),
            pipeline,
        }
    }
}

/// This renders from the intermediate buffer into the output swapchain. It requires HDR
/// alpha to correctly handle the XOR overlapping of SVGs, as the intermediate buffer
/// uses additive blending. Eventually we can use this pass to reuse the rendered shape
/// in multiple places across the image.
///
/// This should probably also be where we handle FXAA
pub mod blit {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    use super::BindId;
    pub use super::Pipeline;

    use memoffset::offset_of;
    use std::mem;

    pub const INTERMEDIATE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    const COLOR_STATES: &[wgpu::ColorStateDescriptor] = &[wgpu::ColorStateDescriptor {
        format: super::SWAPCHAIN_FORMAT,
        color_blend: wgpu::BlendDescriptor {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha_blend: wgpu::BlendDescriptor::REPLACE,
        write_mask: wgpu::ColorWrite::ALL,
    }];

    #[derive(Copy, Clone, PartialEq)]
    pub struct Vertex {
        pub pos: [f32; 2],
        pub uv: [f32; 2],
    }

    unsafe impl bytemuck::Pod for Vertex {}
    unsafe impl bytemuck::Zeroable for Vertex {}

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/blit.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/blit.frag.spv"));
    }

    pub fn build(
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        intermediate_texture: &wgpu::TextureView,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let mut ids = BindId::default();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_blit"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut ids = BindId::default();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: COLOR_STATES,
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, pos) as u64,
                            shader_location: ids.next(),
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, uv) as u64,
                            shader_location: ids.next(),
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let mut ids = BindId::default();
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_blit"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(intermediate_texture),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}
