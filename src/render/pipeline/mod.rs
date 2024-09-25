mod buffer;
mod polyhedron_primitive;
mod texture;

use buffer::Buffer;
use iced::{
    widget::shader::wgpu::{self, RenderPassDepthStencilAttachment},
    Rectangle, Size,
};
use iced_wgpu::wgpu::{DepthBiasState, StencilState};
use iced_winit::core::Color;
use texture::Texture;

pub use buffer::*;
pub use polyhedron_primitive::*;

unsafe impl Send for Scene {}
pub struct Scene {
    pipeline: wgpu::RenderPipeline,
    moment_buf: Buffer,
    shape_buf: Buffer,
    model_buf: Buffer,
    frag_buf: Buffer,
    uniform_group: wgpu::BindGroup,
    depth_texture: Texture,
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        size: &Size<u32>,
    ) -> Scene {
        let pipeline = Self::build_pipeline(device, texture_format);
        // Moment and shape
        let moment_buf = Buffer::new::<MomentVertex>(device, "moment", BufferKind::Vertex);
        let shape_buf = Buffer::new::<ShapeVertex>(device, "shape", BufferKind::Vertex);
        // Create Uniform Buffers
        let model_buf = Buffer::new::<ModelUniforms>(device, "model", BufferKind::Uniform);
        let frag_buf = Buffer::new::<FragUniforms>(device, "frag", BufferKind::Uniform);

        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buf.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frag_buf.raw.as_entire_binding(),
                },
            ],
        });
        let depth_texture = Texture::create_depth_texture(device, size);

        Scene {
            pipeline,
            moment_buf,
            shape_buf,
            model_buf,
            frag_buf,
            uniform_group,
            depth_texture,
        }
    }

    pub fn clear<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
        background_color: Color,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear({
                        let [r, g, b, a] = background_color.into_linear();
                        wgpu::Color {
                            r: r as f64,
                            g: g as f64,
                            b: b as f64,
                            a: a as f64,
                        }
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            // depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            //     view: &self.depth_texture.view,
            //     depth_ops: Some(wgpu::Operations {
            //         load: wgpu::LoadOp::Clear(1.0),
            //         store: wgpu::StoreOp::Store,
            //     }),
            //     stencil_ops: None,
            // }),
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_group, &[]);
        pass.set_vertex_buffer(0, self.moment_buf.raw.slice(..));
        pass.set_vertex_buffer(1, self.shape_buf.raw.slice(..));
        pass.draw(0..self.shape_buf.count, 0..1);
    }

    pub fn update(&mut self) {
        // // Update depth
        // if target_size != self.depth_texture.size {
        //     self.depth_texture = Texture::create_depth_texture(device, &target_size);
        // }
        //
        // if primitive.render.schlegel {
        //     self.starting_vertex = primitive.face_sides_buffer(0).len();
        // } else {
        //     self.starting_vertex = 0;
        // }
        //
        // let moments = primitive.moment_vertices();
        // if self.moment_buf.count != moments.len() as u32 {
        //     self.moment_buf.resize(device, moments.len() as u32);
        //
        //     let shapes = primitive.shape_vertices();
        //     self.shape_buf.resize(device, shapes.len() as u32);
        //     queue.write_buffer(&self.shape_buf.raw, 0, bytemuck::cast_slice(&shapes));
        // }
        //
        // queue.write_buffer(&self.moment_buf.raw, 0, bytemuck::cast_slice(&moments));
        // // Write uniforms
        // self.model_buf.write_data(queue, &uniforms.model);
        // self.frag_buf.write_data(queue, &uniforms.frag);
    }

    fn build_pipeline(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let module = &device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[&uniform_layout],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MomentVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            // position
                            0 => Float32x3,
                            // color
                            1 => Float32x4,
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ShapeVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            // barycentric
                            2 => Float32x4,
                            // sides
                            3 => Float32x4,
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            /* depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }), */
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
}
