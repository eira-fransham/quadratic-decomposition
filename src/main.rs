use std::{iter, time};
use wgpu::util::DeviceExt as _;
use winit::{
    event::{self, DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod cache;
mod pipelines;
mod renderer;

const DEFAULT_SIZE: (u32, u32) = (800, 800);

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    let out_path = if cfg!(debug_assertions) {
        use std::convert::TryFrom;

        Some(
            std::path::PathBuf::try_from(env!("CARGO_MANIFEST_DIR"))
                .unwrap()
                .join("calls.dbg"),
        )
    } else {
        None
    };
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::default(),
                limits: wgpu::Limits::default(),
                shader_validation: cfg!(debug_assertions),
            },
            out_path.as_ref().map(|p| &**p),
        )
        .await
        .unwrap();

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: pipelines::SWAPCHAIN_FORMAT,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let start = time::Instant::now();
    let mut last_update_inst = time::Instant::now();
    let mut last_render_inst = time::Instant::now();

    const FPS: f64 = 60.;
    const DT: f64 = 1. / FPS;
    const TIMEOUT_SECS: f64 = 3.;

    let render_dt = time::Duration::from_secs_f64(DT);

    let mut consecutive_timeouts = 0usize;

    let mut renderer = renderer::Canvas::new(&device, (size.width, size.height));

    let mut mouse_pos = cgmath::Vector2::new(0., 0.);
    let mut mouse_captured = false;
    let mut options = renderer::Options {
        num_segments: std::convert::TryFrom::try_from(2).unwrap(),
    };

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(last_render_inst + render_dt);

        let now = time::Instant::now();

        match event {
            Event::MainEventsCleared => {
                if now - last_render_inst > render_dt {
                    window.request_redraw();
                    last_render_inst = time::Instant::now();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::CursorMoved { position, .. } if mouse_captured => {
                    let size = window.inner_size();
                    mouse_pos = (cgmath::Vector2::new(position.x as f32, position.y as f32)
                        / size.width as f32
                        - cgmath::Vector2::new(0.5, 0.5))
                        * 2.;
                    mouse_pos.y *= -1.;
                }
                WindowEvent::MouseInput {
                    state: event::ElementState::Pressed,
                    button: event::MouseButton::Left,
                    ..
                } => {
                    mouse_captured = !mouse_captured;
                    // TODO: Add path to curve;
                }
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => match keycode {
                    event::VirtualKeyCode::O => {
                        if let Some(val) = options
                            .num_segments
                            .get()
                            .checked_sub(1)
                            .and_then(std::num::NonZeroU8::new)
                        {
                            options.num_segments = val;
                        }
                    }
                    event::VirtualKeyCode::P => {
                        if let Some(val) = options
                            .num_segments
                            .get()
                            .checked_add(1)
                            .and_then(std::num::NonZeroU8::new)
                        {
                            options.num_segments = val;
                        }
                    }
                    _ => {}
                },
                _ => {}
            },
            Event::RedrawRequested(_) => match swap_chain.get_current_frame() {
                Ok(frame) => {
                    consecutive_timeouts = 0;

                    queue.submit(renderer.render(
                        &device,
                        &frame.output.view,
                        [[0.25, 0.], [0.75, 0.75], mouse_pos.into()].iter().copied(),
                        &options,
                    ));
                }
                Err(_) => {
                    consecutive_timeouts += 1;

                    if consecutive_timeouts as f64 > FPS * TIMEOUT_SECS {
                        panic!(
                            "Timeout aquiring swap chain texture ({} consecutive timeouts)",
                            consecutive_timeouts
                        );
                    }
                }
            },
            _ => {}
        }
    });
}

fn main() {
    let events = EventLoop::new();
    let (width, height) = DEFAULT_SIZE;
    let window = WindowBuilder::new()
        .with_resizable(false)
        .with_inner_size(winit::dpi::LogicalSize { width, height })
        .build(&events)
        .unwrap();

    futures::executor::block_on(run(events, window));
}
