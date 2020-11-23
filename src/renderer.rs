use crate::{
    cache::{self, Cache, CacheCommon},
    pipelines::{self, blit, curve},
};
use cgmath::{InnerSpace, Vector2};
use std::num::NonZeroU8;

const MAX_VERTICES_BEFORE_FLUSH: usize = std::u16::MAX as usize;

// TODO: add atlas
pub struct Canvas {
    blit_verts: wgpu::Buffer,
    vertices: cache::BufferCache<pipelines::curve::Vertex>,
    indices: cache::BufferCache<u16>,
    curve_pipeline: curve::Pipeline,
    blit_pipeline: blit::Pipeline,
    intermediate_buffer: wgpu::TextureView,
}

pub struct Options {
    pub num_segments: NonZeroU8,
}

impl Canvas {
    pub fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        use wgpu::util::DeviceExt as _;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blit_intermediate"),
            size: wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: blit::INTERMEDIATE_TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_diffuse"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let intermediate_buffer = texture.create_view(&Default::default());

        let vertices = cache::BufferCache::new(wgpu::BufferUsage::VERTEX);
        let indices = cache::BufferCache::new(wgpu::BufferUsage::INDEX);

        let curve_pipeline = curve::build(device);
        let blit_pipeline = blit::build(device, &sampler, &intermediate_buffer);

        let blit_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                blit::Vertex {
                    pos: [-1., -1.],
                    uv: [0., 1.],
                },
                blit::Vertex {
                    pos: [1., -1.],
                    uv: [1., 1.],
                },
                blit::Vertex {
                    pos: [1., 1.],
                    uv: [1., 0.],
                },
                blit::Vertex {
                    pos: [1., 1.],
                    uv: [1., 0.],
                },
                blit::Vertex {
                    pos: [-1., 1.],
                    uv: [0., 0.],
                },
                blit::Vertex {
                    pos: [-1., -1.],
                    uv: [0., 1.],
                },
            ]),
            usage: wgpu::BufferUsage::VERTEX,
        });
        Self {
            vertices,
            indices,
            curve_pipeline,
            blit_pipeline,
            intermediate_buffer,
            blit_verts,
        }
    }

    pub fn render<V>(
        &mut self,
        device: &wgpu::Device,
        screen_tex: &wgpu::TextureView,
        // TODO: Make this "paths" plural and add some kind of separation so we can independently
        //       cache the paths.
        path: impl IntoIterator<Item = V>,
        options: &Options,
    ) -> Option<wgpu::CommandBuffer>
    where
        V: Into<Vector2<f32>>,
    {
        use itertools::Itertools;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render"),
        });

        // Specified in screen coordinates. This code assumes a square screen right now but
        // extending it to support arbitrary aspect ratios would be relatively trivial.
        let width: f32 = 0.05;

        self.vertices.clear();
        self.indices.clear();

        let num_segments = options.num_segments.get() as u32;
        let num_points = (num_segments * 2) - 1;
        let dt = ((num_points + 1) as f32).recip();

        let verts = self.vertices.append(
            path.into_iter()
                .map(Into::into)
                .tuple_windows()
                .step_by(2)
                .flat_map(|(start, control, end)| {
                    /// Calculates a quadratic bezier that passes through the start and end
                    /// point at t=0 and t=1 respectively, and passes through the midpoint
                    /// at t=midpoint_t. It doesn't take gradient into account, which leads
                    /// to discontinuities and should be addressed.
                    fn calc_quadratic(
                        start: Vector2<f32>,
                        mid: Vector2<f32>,
                        end: Vector2<f32>,
                        midpoint_t: f32,
                    ) -> (Vector2<f32>, Vector2<f32>, Vector2<f32>) {
                        let t = midpoint_t;

                        (
                            start,
                            (mid - t.powi(2) * end - (1. - t).powi(2) * start)
                                / (2. * t * (1. - t)),
                            end,
                        )
                    }

                    fn point_on_quadratic(
                        start: Vector2<f32>,
                        control: Vector2<f32>,
                        end: Vector2<f32>,
                        t: f32,
                    ) -> Vector2<f32> {
                        start * (1. - t).powi(2) + 2. * control * t * (1. - t) + end * t.powi(2)
                    }

                    fn normal_at(
                        start: Vector2<f32>,
                        control: Vector2<f32>,
                        end: Vector2<f32>,
                        t: f32,
                    ) -> Vector2<f32> {
                        // Basic derivation of bezier formula
                        let tangent = (2. * (1. - t) * (control - start)
                            + 2. * t * (end - control))
                            .normalize();

                        // Rotate 90 degrees
                        Vector2::from((-tangent.y, tangent.x))
                    }

                    /// Triangulate a quad. This function is extremely bad and
                    /// should handle automatically generating correct vertices
                    /// for both clockwise and counter-clockwise quads.
                    fn quad(input: [Vector2<f32>; 4]) -> [Vector2<f32>; 6] {
                        [input[0], input[2], input[3], input[3], input[1], input[0]]
                    }

                    // TODO: This only works for counterclockwise curves, but it shouldn't be too
                    //       difficult to make it work for clockwise curves. This would probably
                    //       make it look better for extreme angles, too, as these issues seem
                    //       to be connected.
                    (0..num_points)
                        // We step by 2 but triangulate 3 points from each step, as the end of 1 curve
                        // is the start of the next but the control points are not shared. The points
                        // look like:
                        //
                        // [start1, control1, end1/start2, control2, end2/start3, ..]
                        .step_by(2)
                        .flat_map(|i| {
                            let i = i as f32 * dt;

                            let (a, b, c) = (i, i + dt, i + dt * 2.);

                            let (a_point, b_point, c_point) = (
                                point_on_quadratic(start, control, end, a),
                                point_on_quadratic(start, control, end, b),
                                point_on_quadratic(start, control, end, c),
                            );
                            let (a_norm, b_norm, c_norm) = (
                                normal_at(start, control, end, a),
                                normal_at(start, control, end, b),
                                normal_at(start, control, end, c),
                            );

                            // TODO: This leads to weird C1 discontinuities, so we might want to solve for
                            //       gradient at the start and end and minimise positional error, or have
                            //       some other way to normalise the gradient between segments to maintain
                            //       C1 continuity.
                            let perc = (b - a) / (c - a);
                            let inner = calc_quadratic(
                                a_point + a_norm * width,
                                b_point + b_norm * width,
                                c_point + c_norm * width,
                                perc,
                            );

                            let outer = calc_quadratic(
                                a_point - a_norm * width,
                                b_point - b_norm * width,
                                c_point - c_norm * width,
                                perc,
                            );

                            let mut out = vec![
                                curve::Vertex {
                                    pos: outer.0.into(),
                                    uv: [0., 0.],
                                    sign: 1.,
                                },
                                curve::Vertex {
                                    pos: outer.1.into(),
                                    uv: [0.5, 0.],
                                    sign: 1.,
                                },
                                curve::Vertex {
                                    pos: outer.2.into(),
                                    uv: [1., 1.],
                                    sign: 1.,
                                },
                                curve::Vertex {
                                    pos: inner.0.into(),
                                    uv: [0., 0.],
                                    sign: -1.,
                                },
                                curve::Vertex {
                                    pos: inner.1.into(),
                                    uv: [0.5, 0.],
                                    sign: -1.,
                                },
                                curve::Vertex {
                                    pos: inner.2.into(),
                                    uv: [1., 1.],
                                    sign: -1.,
                                },
                            ];

                            out.extend(
                                quad([
                                    inner.0,
                                    inner.1,
                                    outer.0,
                                    point_on_quadratic(outer.0, outer.1, outer.2, 0.5),
                                ])
                                .iter()
                                .copied()
                                .map(|p| curve::Vertex {
                                    pos: p.into(),
                                    uv: Default::default(),
                                    sign: 0.,
                                }),
                            );

                            out.extend(
                                quad([
                                    inner.1,
                                    inner.2,
                                    point_on_quadratic(outer.0, outer.1, outer.2, 0.5),
                                    outer.2,
                                ])
                                .iter()
                                .copied()
                                .map(|p| curve::Vertex {
                                    pos: p.into(),
                                    uv: Default::default(),
                                    sign: 0.,
                                }),
                            );

                            out
                        })
                        .collect::<Vec<_>>()
                }),
        );

        self.indices.append(verts.start as u16..verts.end as u16);

        self.vertices.update(device, &mut encoder);
        self.indices.update(device, &mut encoder);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &self.intermediate_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.,
                            g: 0.,
                            b: 0.,
                            a: 0.,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.curve_pipeline.pipeline);
            rpass.set_vertex_buffer(0, self.vertices.as_ref()?.slice(..));
            rpass.set_index_buffer(self.indices.as_ref()?.slice(..));
            rpass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: screen_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.,
                            g: 0.,
                            b: 0.,
                            a: 0.,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.blit_pipeline.pipeline);
            rpass.set_bind_group(0, &self.blit_pipeline.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.blit_verts.slice(..));
            rpass.draw(0..6, 0..1)
        }

        Some(encoder.finish())
    }
}
