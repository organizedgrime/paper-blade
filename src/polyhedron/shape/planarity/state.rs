use std::{collections::HashMap, hash::Hash};

use crate::polyhedron::{shape::Distance, VertexId, Edge};

pub struct LRState {
    graph: Distance,
    roots: Vec<VertexId>,
    height: HashMap<VertexId, usize>,
    edge_parent: HashMap<VertexId, Edge>,
    low_point: HashMap<Edge, usize>,
    low_point_2: HashMap<Edge, usize>,
    low_point_edge: HashMap<Edge, Edge>,
    nesting_depth: HashMap<Edge, usize>,
    //stack: Vec<ConflictPair<Edge>>,
    //stack_emarker: HashMap<Edge, ConflictPair<Edge>>,
    eref: HashMap<Edge, Edge>,
    /// side of edge, or modifier for side of reference edge.
    side: HashMap<Edge, Sign>,
}

impl LRState {
    pub fn new(graph: Distance) -> Self {
        let v = graph.order();
        let e = graph.edges().count();
        let side = graph.edges().map(|e| (e.into(), Sign::Plus)).collect();
        Self {
            graph,
            roots: Vec::new(),
            height: HashMap::with_capacity(v),
            edge_parent: HashMap::with_capacity(e),
            low_point: HashMap::with_capacity(e),
            low_point_2: HashMap::with_capacity(e),
            low_point_edge: HashMap::with_capacity(e),
            nesting_depth: HashMap::with_capacity(e),
            eref: HashMap::with_capacity(e),
            side,
        }
    }
}

enum Sign {
    Plus,
    Minus
}

