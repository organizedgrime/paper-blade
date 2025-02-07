mod conway;
mod platonic;
mod svg;

#[cfg(test)]
mod test;

use crate::polyhedron::VertexId;
use std::{
    fmt::Display,
    ops::{Index, IndexMut, Range},
};

/// Jagged array which represents the symmetrix distance matrix of a given Graph
/// usize::MAX    ->   disconnected
/// 0             ->   identity
/// n             ->   actual distance
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct Distance {
    /// The order is the number of vertices
    order: usize,
    distance: Vec<usize>,
}

impl Distance {
    /// [ 0 ]
    /// [ M | 0 ]
    /// [ M | M | 0 ]
    /// ..
    /// [ M | M | M | ... | M | M | M | 0 ]
    pub fn new(n: usize) -> Self {
        Distance {
            order: n,
            distance: (0..n)
                .flat_map(|m| [vec![usize::MAX; m], vec![0]].concat())
                .collect(),
        }
    }
}

impl Distance {
    /// Connect one vertex to another with length one, iff they are note the same point
    pub fn connect(&mut self, [v, u]: [VertexId; 2]) {
        if self[[v, u]] != 0 {
            self[[v, u]] = 1;
        }
    }

    /// Disconnect one vertex from another iff they are neighbors
    pub fn disconnect(&mut self, [v, u]: [VertexId; 2]) {
        if self[[v, u]] == 1 {
            self[[v, u]] = usize::MAX;
        }
    }

    /// Inserts a new vertex in the matrix
    pub fn insert(&mut self) -> VertexId {
        self.distance
            .extend([vec![usize::MAX; self.order], vec![0]].concat());
        self.order += 1;
        self.order - 1
    }

    /// Deletes a vertex from the matrix
    pub fn delete(&mut self, v: VertexId) {
        let mut distance = Distance::new(self.order - 1);
        for i in 0..self.order {
            for j in i..self.order {
                if i != v && j != v {
                    let x = if i > v { i - 1 } else { i };
                    let y = if j > v { j - 1 } else { j };
                    distance[[x, y]] = self[[i, j]];
                }
            }
        }
        *self = distance;
    }

    /// Enumerates the vertices connected to v
    pub fn neighbors(&self, v: VertexId) -> Vec<VertexId> {
        self.vertices().filter(|&u| self[[v, u]] == 1).collect()
    }

    /// Iterable Range representing vertex IDs
    pub fn vertices(&self) -> Range<VertexId> {
        0..self.order
    }

    /// All possible compbinations of vertices
    pub fn vertex_pairs(&self) -> impl Iterator<Item = [VertexId; 2]> {
        self.vertices().flat_map(|v| (0..v).map(move |u| [v, u]))
    }

    /// All actual edges of the graph (D_{ij} = 1)
    pub fn edges(&self) -> impl Iterator<Item = [VertexId; 2]> + use<'_> {
        self.vertex_pairs().filter(move |&e| self[e] == 1)
    }

    /// Vertex Count
    pub fn order(&self) -> usize {
        self.order
    }

    /// Maximum distance value
    pub fn diameter(&self) -> usize {
        self.vertex_pairs().map(|e| self[e]).max().unwrap_or(0)
    }

    /* fn dfs(&self, visited: &mut HashSet<usize>, v: usize) {
        visited.insert(v);
        for u in self.neighbors(v) {
            if !visited.contains(&u) {
                self.dfs(visited, u);
            }
        }
    }

    fn is_connected(&self) -> bool {
        let mut visited = HashSet::new();
        self.dfs(&mut visited, 0);
        visited.len() == self.order()
    } */

    /// This functiona and the helpers it relies on should soon be superceded by
    /// a proper plane embedding.
    pub fn cycle_is_face(&self, mut cycle: Vec<VertexId>) -> bool {
        let mut dupe = self.clone();
        while !cycle.is_empty() {
            let v = cycle.remove(0);
            dupe.delete(v);
            for u in &mut cycle {
                if *u > v {
                    *u -= 1;
                }
            }
        }

        // dupe.is_connected()
        dupe.is_planar()
    }

    /// Use a simple BFS to compute the shortest paths for all pairs
    pub fn bfs_apsp(&mut self) {
        for source in self.vertices() {
            let mut visited = vec![false; self.order()];
            let mut queue = vec![source];
            visited[source] = true;
            while !queue.is_empty() {
                let current = queue.remove(0);
                for neighbor in self.neighbors(current) {
                    if !visited[neighbor] {
                        self[[source, neighbor]] = self[[source, current]] + 1;
                        queue.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }

    pub fn springs(&self) -> Vec<[VertexId; 2]> {
        let diameter = self.diameter();
        self.vertex_pairs()
            .filter(|&[v, u]| v != u && (self[[v, u]] <= 2 || self[[v, u]] >= diameter - 1))
            .collect::<Vec<_>>()
    }

    /// Number of faces
    pub fn face_count(&self) -> i64 {
        2 + self.edges().count() as i64 - self.order() as i64
    }
}

impl Index<[VertexId; 2]> for Distance {
    type Output = usize;

    fn index(&self, index: [VertexId; 2]) -> &Self::Output {
        let x = index[0].max(index[1]);
        let y = index[0].min(index[1]);
        &self.distance[(x * (x + 1)) / 2 + y]
    }
}

impl IndexMut<[VertexId; 2]> for Distance {
    fn index_mut(&mut self, index: [VertexId; 2]) -> &mut Self::Output {
        let x = index[0].max(index[1]);
        let y = index[0].min(index[1]);
        &mut self.distance[(x * (x + 1)) / 2 + y]
    }
}

impl Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("\t|"))?;
        for i in 0..self.order() {
            f.write_fmt(format_args!(" {i} |"))?;
        }
        f.write_fmt(format_args!("\n\t"))?;
        for _ in 0..self.order() {
            f.write_fmt(format_args!("____"))?;
        }
        f.write_fmt(format_args!("\n"))?;
        for v in self.vertices() {
            f.write_fmt(format_args!("{v}:\t|"))?;
            for u in self.vertices() {
                let value = if self[[v, u]] == usize::MAX {
                    String::from("_")
                } else {
                    self[[v, u]].to_string()
                };
                f.write_fmt(format_args!(" {value} |"))?;
            }
            f.write_fmt(format_args!("\n"))?;
        }
        Ok(())
    }
}
