use crate::{bones::PolyGraph, render::camera::Camera};
use iced::widget::shader::{self, wgpu};
use iced::{Color, Rectangle, Size};
use ultraviolet::{Mat4, Vec3, Vec4};

use super::{AllUniforms, FragUniforms, LightUniforms, ModelUniforms, Pipeline, Vertex};

#[derive(Debug)]
pub struct PolyhedronPrimitive {
    polyhedron: PolyGraph,
    palette: Vec<wgpu::Color>,
    transform: Mat4,
    camera: Camera,
}

impl PolyhedronPrimitive {
    pub fn new(
        polyhedron: PolyGraph,
        palette: Vec<wgpu::Color>,
        transform: Mat4,
        camera: Camera,
    ) -> Self {
        Self {
            polyhedron,
            palette,
            transform,
            camera,
        }
    }

    fn face_triangle_positions(&self, face_index: usize) -> Vec<Vec3> {
        let positions = self.polyhedron.face_positions(face_index);
        let n = positions.len();
        match n {
            3 => positions,
            4 => vec![
                positions[0],
                positions[1],
                positions[2],
                positions[2],
                positions[3],
                positions[0],
            ],
            _ => {
                let centroid = self.polyhedron.face_centroid(face_index);
                let n = positions.len();
                (0..n).fold(vec![], |acc, i| {
                    [acc, vec![positions[i], centroid, positions[(i + 1) % n]]].concat()
                })
            }
        }
    }

    pub fn positions(&self) -> Vec<Vec3> {
        (0..self.polyhedron.cycles.len()).fold(Vec::new(), |acc, i| {
            [acc, self.face_triangle_positions(i)].concat()
        })
    }

    fn face_sides_buffer(&self, face_index: usize) -> Vec<Vec3> {
        let positions = self.polyhedron.face_positions(face_index);
        let n = positions.len();
        match n {
            3 => vec![Vec3::new(1.0, 1.0, 1.0); 3],
            4 => vec![Vec3::new(1.0, 0.0, 1.0); 6],
            _ => vec![Vec3::new(0.0, 1.0, 0.0); n * 3],
        }
    }

    pub fn vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        let barycentric = [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()];

        let mut polygon_sizes: Vec<usize> =
            self.polyhedron
                .cycles
                .iter()
                .fold(Vec::new(), |mut acc, f| {
                    if !acc.contains(&f.len()) {
                        acc.push(f.len());
                    }
                    acc
                });

        polygon_sizes.sort();

        for i in 0..self.polyhedron.cycles.len() {
            let color_index = polygon_sizes
                .iter()
                .position(|&x| x == self.polyhedron.cycles[i].len())
                .unwrap();

            let n = polygon_sizes.get(color_index).unwrap();
            let color = self.palette[n % self.palette.len()];
            let color = Vec4::new(color.r as f32, color.g as f32, color.b as f32, 1.0);
            let sides = self.face_sides_buffer(i);
            let positions = self.face_triangle_positions(i);

            for j in 0..positions.len() {
                let p = positions[j].normalized();
                let b = barycentric[j % barycentric.len()];
                vertices.push(Vertex {
                    normal: Vec4::new(p.x, p.y, p.z, 0.0),
                    sides: Vec4::new(sides[j].x, sides[j].y, sides[j].z, 0.0),
                    barycentric: Vec4::new(b.x, b.y, b.z, 0.0),
                    color,
                });
            }
        }

        vertices
    }
}

impl shader::Primitive for PolyhedronPrimitive {
    fn prepare(
        &self,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: Rectangle,
        target_size: Size<u32>,
        _scale_factor: f32,
        storage: &mut shader::Storage,
    ) {
        let vertex_count = self.polyhedron.vertex_count();
        if !storage.has::<Pipeline>() {
            storage.store(Pipeline::new(device, format, target_size, vertex_count));
        }
        let pipeline = storage.get_mut::<Pipeline>().unwrap();

        // update uniform buffer
        let model_mat = self.transform;
        let view_projection_mat = self.camera.build_view_proj_mat(bounds);
        let uniforms = AllUniforms {
            model: ModelUniforms {
                model_mat,
                view_projection_mat,
            },
            frag: FragUniforms {
                light_position: self.camera.position(),
                eye_position: self.camera.position() + Vec4::new(2.0, 2.0, 1.0, 0.0),
            },
            light: LightUniforms::new(
                Color::new(1.0, 1.0, 1.0, 1.0),
                Color::new(1.0, 1.0, 1.0, 1.0),
            ),
        };

        //upload data to GPU
        pipeline.update(device, queue, target_size, vertex_count, &uniforms, self);
    }

    fn render(
        &self,
        storage: &shader::Storage,
        target: &wgpu::TextureView,
        _target_size: Size<u32>,
        viewport: Rectangle<u32>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // At this point our pipeline should always be initialized
        let pipeline = storage.get::<Pipeline>().unwrap();

        // Render primitive
        pipeline.render(target, encoder, viewport);
    }
}
