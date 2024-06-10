use crate::scene::Vertex;

use super::*;
use glam::{vec3, Vec3};
use std::collections::{HashMap, HashSet};

const TICK_SPEED: f32 = 200.0;

// Operations
impl PolyGraph {
    fn apply_spring_forces(&mut self) {
        let diam = *self.dist.values().max().unwrap_or(&1) as f32;
        let l_diam = self.edge_length * 2.0;
        for v in self.vertices.iter() {
            for u in self.vertices.iter() {
                if u != v {
                    let e: Edge = (v, u).into();
                    let d = self.dist[&e] as f32;
                    let l = l_diam * (d / diam);
                    let k = 1.0 / d;
                    if self.contracting_edges.contains(&e) {
                        let v_position = self.positions[v];
                        let u_position = self.positions[u];
                        let l = v_position.distance(u_position);
                        let f = (self.edge_length / TICK_SPEED * 3.0) / l;
                        *self.positions.get_mut(v).unwrap() = v_position.lerp(u_position, f);
                        *self.positions.get_mut(u).unwrap() = u_position.lerp(v_position, f);
                    } else {
                        let diff = self.positions[v] - self.positions[u];
                        let dist = diff.length();
                        let distention = l - dist;
                        let restorative_force = k / 2.0 * distention;
                        let f = diff * restorative_force / TICK_SPEED;

                        let v_speed = self.speeds.get_mut(v).unwrap();
                        *v_speed += f;
                        *v_speed *= 0.92;
                        *self.positions.get_mut(v).unwrap() += *v_speed;

                        let u_speed = self.speeds.get_mut(u).unwrap();
                        *u_speed -= f;
                        *u_speed *= 0.92;
                        *self.positions.get_mut(u).unwrap() += *u_speed;
                    }
                }
            }
        }
    }

    fn center(&mut self) {
        let shift =
            self.positions.values().fold(Vec3::ZERO, |a, &b| a + b) / self.vertices.len() as f32;

        for (_, v) in self.positions.iter_mut() {
            *v -= shift;
        }
    }

    fn resize(&mut self) {
        let mean_length = self
            .positions
            .values()
            .map(|p| p.length())
            .fold(0.0, f32::max);
        let distance = mean_length - 1.0;
        self.edge_length -= distance / TICK_SPEED;
    }

    pub fn update(&mut self) {
        self.center();
        self.resize();
        self.animate_contraction();
        self.apply_spring_forces();
    }

    fn face_positions(&self, face_index: usize) -> Vec<Vec3> {
        self.faces[face_index]
            .iter()
            .map(|v| self.positions[v])
            .collect()
    }

    fn face_centroid(&self, face_index: usize) -> Vec3 {
        // All vertices associated with this face
        let vertices: Vec<_> = self.face_positions(face_index);
        vertices.iter().fold(Vec3::ZERO, |a, &b| a + b) / vertices.len() as f32
    }

    fn face_triangle_positions(&self, face_index: usize) -> Vec<Vec3> {
        let positions = self.face_positions(face_index);
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
                let centroid = self.face_centroid(face_index);
                let n = positions.len();
                (0..n).fold(vec![], |acc, i| {
                    [acc, vec![positions[i], centroid, positions[(i + 1) % n]]].concat()
                })
            }
        }
    }

    fn face_sides_buffer(&self, face_index: usize) -> Vec<Vec3> {
        let positions = self.face_positions(face_index);
        let n = positions.len();
        match n {
            3 => vec![vec3(1.0, 1.0, 1.0); 3],
            4 => vec![vec3(1.0, 0.0, 1.0); 6],
            _ => vec![vec3(0.0, 1.0, 0.0); n * 3],
        }
    }

    pub fn positions(&self) -> Vec<Vec3> {
        (0..self.faces.len()).fold(Vec::new(), |acc, i| {
            [acc, self.face_triangle_positions(i)].concat()
        })
    }

    pub fn poly_color(n: &usize) -> Vec3 {
        let colors = [
            vec3(72.0, 132.0, 90.0),
            vec3(163.0, 186.0, 112.0),
            vec3(51.0, 81.0, 69.0),
            vec3(254.0, 240.0, 134.0),
            vec3(95.0, 155.0, 252.0),
            vec3(244.0, 164.0, 231.0),
            vec3(170.0, 137.0, 190.0),
        ];

        colors[n % colors.len()] / 255.0
    }

    pub fn vertex_triangle_count(&self) -> u64 {
        let mut count = 0;
        for face in self.faces.iter() {
            match face.len() {
                3 => {
                    count += 3;
                }
                4 => {
                    count += 6;
                }
                _ => {
                    count += 3 * face.len() as u64;
                }
            }
        }
        count
    }

    pub fn vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        let barycentric = [Vec3::X, Vec3::Y, Vec3::Z];

        let color_indices = self.faces.iter().fold(HashMap::new(), |mut acc, f| {
            if !acc.contains_key(&f.len()) {
                acc.insert(f.len(), acc.len());
            }
            acc
        });

        for i in 0..self.faces.len() {
            let color = Self::poly_color(color_indices.get(&self.faces[i].len()).unwrap());
            let sides = self.face_sides_buffer(i);
            let positions = self.face_triangle_positions(i);

            for j in 0..positions.len() {
                vertices.push(Vertex {
                    normal: positions[j].normalize(),
                    sides: sides[j],
                    barycentric: barycentric[j % barycentric.len()],
                    color,
                });
            }
        }

        vertices
    }

    pub fn vertex_buffer_size(&self) -> u64 {
        std::mem::size_of::<Vertex>() as u64 * self.vertex_triangle_count()
    }

    pub fn position_buffer_size(&self) -> u64 {
        std::mem::size_of::<Vec3>() as u64 * self.vertex_triangle_count()
    }

    pub fn animate_contraction(&mut self) {
        // If all edges are contracted visually
        if !self.contracting_edges.is_empty()
            && self.contracting_edges.iter().fold(true, |acc, e| {
                if self.positions.contains_key(&e.v()) && self.positions.contains_key(&e.u()) {
                    acc && self.positions[&e.v()].distance(self.positions[&e.u()]) < 0.08
                } else {
                    acc
                }
            })
        {
            // Contract them in the graph
            for e in self.contracting_edges.clone() {
                self.contract_edge(e);
            }
            self.contracting_edges = HashSet::new();
            self.pst();
            self.name.truncate(self.name.len() - 1);
            self.name += "a";
        }
    }
}
