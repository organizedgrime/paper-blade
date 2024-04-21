use std::collections::{HashMap, HashSet};

pub use super::*;

impl PolyGraph {
    pub fn contract_edge(&mut self, e: impl Into<Edge>) {
        let e: Edge = e.into();
        // If this is the ghost edge, find its extant counterpart
        let id = self.ghost_edges.get(&e).unwrap_or(&e).id();
        // Give b all the same connections as a
        for b in self.connections(&id.0).iter() {
            if b != &id.1 {
                self.connect((b, &id.1))
            }
        }
        // Delete a
        self.delete(&id.0);
        for (_, v) in self.ghost_edges.iter_mut() {
            if let Some(u) = v.other(&id.0) {
                *v = (id.1, u).into();
            }
        }
    }

    pub fn split_vertex(&mut self, v: &VertexId) {
        let original_position = self.positions[v];
        let mut connections: HashSet<usize> = self.connections(v);
        connections.extend(self.ghost_connections(v));
        connections.remove(v);
        let n = connections.len();
        // Previously processed connection, starts with a seed
        let mut previous = *connections.iter().collect::<Vec<_>>()[0];
        let mut new_vertex = 0;

        // Remove the vertex
        self.delete(v);
        // Recompute distances in the absence of the vertex
        self.distances();
        self.faces();

        'connections: while !connections.is_empty() {
            // closest vertex to the previous which is not itself and is connected
            let u = connections.clone().into_iter().collect::<Vec<_>>()[0];

            // Insert a new node in the same location
            new_vertex = self.insert();
            self.positions.insert(new_vertex, original_position);
            // Reform old connection
            self.connect((u, new_vertex));

            // Track
            previous = u;
            connections.remove(&u);

            // Track the ghost edge and new edge
            let ge: Edge = (*v, u).into();
            let ne: Edge = (new_vertex, u).into();

            // If the ghost of this transaction was the new edge of a previous transaction
            for (_, v) in self.ghost_edges.iter_mut() {
                if v.id() == ge.id() {
                    // Update its child
                    *v = ne;
                    continue 'connections;
                }
            }
            // Track ghost edge directly if one didnt already exist
            self.ghost_edges.insert(ge, ne);
        }

        // Connect all nodes in the new face formed
        for i in 0..n - 1 {
            self.connect((new_vertex - i, new_vertex - i - 1));
        }
        self.connect((new_vertex, new_vertex - n + 1));
    }

    /// `t` truncate
    pub fn truncate(&mut self) {
        for v in self.vertices.clone().iter() {
            self.split_vertex(v);
        }
        self.recompute_qualities();
        self.name += "t";
    }

    /// `a` ambo
    pub fn ambo(&mut self) {
        let original_edges = self.adjacents.clone();
        // Truncate
        self.truncate();

        //self.contract_edges_visually(original_edges);
        // Animate

        //self.contracting_edges.extend(original_edges);
        // Contract original edge set
        for edge in original_edges.iter() {
            self.contract_edge(*edge);
        }

        self.recompute_qualities();
        self.ghost_edges = HashMap::new();
        self.name.truncate(self.name.len() - 1);
        self.name += "a";
    }

    /// `b` = `ta`
    pub fn bevel(&mut self) {
        self.truncate();
        self.ambo();
        self.name.truncate(self.name.len() - 2);
        self.name += "b";
    }

    /// `e` = `aa`
    pub fn expand(&mut self) {
        self.ambo();
        self.ambo();
        self.name.truncate(self.name.len() - 2);
        self.name += "e";
    }

    /// `s` snub is applying `e` followed by diagonal addition
    pub fn snub(&mut self) {
        self.expand();
        //self.diagonal_addition();
    }

    // `j` join
    // `z` zip
    // `g` gyro
    // `m` meta = `kj`
    // `o` ortho = `jj`
    // `n` needle
    // `k` kis
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn truncate() {
        let mut shape = PolyGraph::icosahedron();
        shape.truncate();
    }

    #[test]
    fn contract_edge() {
        let mut graph = PolyGraph::new_disconnected(6);
        graph.connect((1, 0));
        graph.connect((1, 2));

        graph.connect((1, 3));

        graph.connect((3, 4));
        graph.connect((3, 5));
        graph.recompute_qualities();

        assert_eq!(graph.vertices.len(), 6);
        assert_eq!(graph.adjacents.len(), 5);

        graph.contract_edge((1, 3));
        graph.recompute_qualities();

        assert_eq!(graph.vertices.len(), 5);
        println!("adja: {:?}", graph.adjacents);
        assert_eq!(graph.adjacents.len(), 4);

        assert_eq!(graph.connections(&0), vec![3].into_iter().collect());
        assert_eq!(graph.connections(&2), vec![3].into_iter().collect());

        assert_eq!(
            graph.connections(&3),
            vec![0, 2, 4, 5].into_iter().collect()
        );

        assert_eq!(graph.connections(&4), vec![3].into_iter().collect());
        assert_eq!(graph.connections(&5), vec![3].into_iter().collect());
    }

    #[test]
    fn split_vertex() {
        let mut graph = PolyGraph::new_disconnected(5);
        graph.connect((1, 0));
        graph.connect((1, 2));

        graph.connect((1, 3));
        graph.connect((1, 4));
        graph.recompute_qualities();

        assert_eq!(graph.vertices.len(), 5);
        assert_eq!(graph.adjacents.len(), 4);

        graph.split_vertex(&1);
        graph.recompute_qualities();

        assert_eq!(graph.vertices.len(), 8);
        assert_eq!(graph.adjacents.len(), 8);
    }
}
