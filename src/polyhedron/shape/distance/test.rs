use std::fs::create_dir_all;

use super::*;
use crate::render::message::PresetMessage::{self, *};
use test_case::test_case;

impl Distance {
    pub fn floyd(&mut self) {
        // let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
        let mut graph: Distance = Distance::new(self.distance.len());
        for e in self.edges() {
            graph[e] = 1;
        }
        for k in graph.vertices() {
            for i in graph.vertices() {
                for j in graph.vertices() {
                    if graph[[i, k]] != usize::MAX && graph[[k, j]] != usize::MAX {
                        let nv = graph[[i, k]] + graph[[k, j]];
                        if graph[[i, j]] > nv || graph[[j, i]] > nv {
                            graph[[i, j]] = nv;
                        }
                    }
                }
            }
        }
        *self = graph;
    }

    /// Hardcoded Tetrahedron construction to isolate testing
    pub fn tetrahedron() -> Self {
        let mut tetra = Distance::new(4);
        tetra[[0, 1]] = 1;
        tetra[[0, 2]] = 1;
        tetra[[0, 3]] = 1;
        tetra[[1, 2]] = 1;
        tetra[[1, 3]] = 1;
        tetra[[2, 3]] = 1;
        tetra
    }
}

#[test]
fn basics() {
    let mut graph = Distance::new(4);
    println!("basics:");
    // Connect
    graph.connect([0, 1]);
    graph.connect([0, 2]);
    graph.connect([1, 2]);
    assert_eq!(graph.connections(0), vec![1, 2]);
    assert_eq!(graph.connections(1), vec![0, 2]);
    assert_eq!(graph.connections(2), vec![0, 1]);
    assert_eq!(graph.connections(3), vec![]);

    // Disconnect
    graph.disconnect([0, 1]);
    assert_eq!(graph.connections(0), vec![2]);
    assert_eq!(graph.connections(1), vec![2]);

    // Delete
    graph.delete(1);
    assert_eq!(graph.connections(0), vec![1]);
    assert_eq!(graph.connections(2), vec![]);
    assert_eq!(graph.connections(1), vec![0]);
}

#[test]
fn chordless_cycles() {
    let mut graph = Distance::new(4);
    // Connect
    graph.connect([0, 1]);
    graph.connect([1, 2]);
    graph.connect([2, 3]);

    println!("chordless_cycles:");
    println!("{graph}");
    graph.pst();
    println!("{graph}");

    graph.connect([2, 0]);
}

#[test]
fn contract_edge() {
    let mut graph = Distance::tetrahedron();
    graph.contract_edge([0, 2]);
    let mut triangle = Distance::new(3);
    triangle[[0, 1]] = 1;
    triangle[[1, 2]] = 1;
    triangle[[2, 0]] = 1;
    assert_eq!(graph, triangle);
}