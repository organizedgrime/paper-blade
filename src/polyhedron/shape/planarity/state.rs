use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use crate::polyhedron::{shape::Distance, Edge, VertexId};

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

pub type Time = usize;

#[derive(Copy, Clone, Debug)]
pub enum DfsEvent {
    Discover(VertexId, Time),
    /// An edge of the tree formed by the traversal.
    TreeEdge(VertexId, VertexId),
    /// An edge to an already visited node.
    BackEdge(VertexId, VertexId),
    /// A cross or forward edge.
    ///
    /// For an edge *(u, v)*, if the discover time of *v* is greater than *u*,
    /// then it is a forward edge, else a cross edge.
    CrossForwardEdge(VertexId, VertexId),
    /// All edges from a node have been reported.
    Finish(VertexId, Time),
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

    pub fn lr_orientation_visitor(&mut self, event: DfsEvent) {
        match event {
            DfsEvent::Discover(v, _) => {
                if let Entry::Vacant(entry) = self.height.entry(v) {
                    entry.insert(0);
                    self.roots.push(v);
                }
            }
            DfsEvent::TreeEdge(v, w) => {
                let ei = [v, w].into();
                let v_height = self.height[&v];
                let w_height = v_height + 1;

                self.edge_parent.insert(w, ei);
                self.height.insert(w, w_height);
                // now initialize low points.
                self.low_point.insert(ei, v_height);
                self.low_point_2.insert(ei, w_height);
            }
            DfsEvent::BackEdge(v, w) => {
                // do *not* consider ``(v, w)`` as a back edge if ``(w, v)`` is a tree edge.
                let edge = [v, w].into();
                if Some(&edge) != self.edge_parent.get(&v) {
                    self.low_point.insert(edge, self.height[&w]);
                    self.low_point_2.insert(edge, self.height[&v]);
                }
            }
            DfsEvent::Finish(v, _) => {
                for w in self.graph.neighbors(v) {
                    //let ei = (v, w);
                    let ei: Edge = [v, w].into();

                    // determine nesting depth.
                    let low = match self.low_point.get(&ei) {
                        Some(val) => *val,
                        None =>
                        // if ``lowpt`` does *not* contain edge ``(v, w)``, it means
                        // that it's *not* a tree or a back edge so we skip it since
                        // it's oriented in the reverse direction.
                        {
                            continue
                        }
                    };

                    if self.low_point_2[&ei] < self.height[&v] {
                        // if it's chordal, add one.
                        self.nesting_depth.insert(ei, 2 * low + 1);
                    } else {
                        self.nesting_depth.insert(ei, 2 * low);
                    }

                    // update lowpoints of parent edge.
                    if let Some(e_par) = self.edge_parent.get(&v) {
                        match self.low_point[&ei].cmp(&self.low_point[e_par]) {
                            Ordering::Less => {
                                self.low_point_2.insert(
                                    *e_par,
                                    self.low_point[e_par].min(self.low_point_2[&ei]),
                                );
                                self.low_point.insert(*e_par, self.low_point[&ei]);
                            }
                            Ordering::Greater => {
                                modify_if_min(&mut self.low_point_2, *e_par, self.low_point[&ei]);
                            }
                            _ => {
                                let val = self.low_point_2[&ei];
                                modify_if_min(&mut self.low_point_2, *e_par, val);
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }
}

fn modify_if_min<K, V>(xs: &mut HashMap<K, V>, key: K, val: V)
where
    K: Hash + Eq,
    V: Ord + Copy,
{
    xs.entry(key).and_modify(|e| {
        if val < *e {
            *e = val;
        }
    });
}

enum Sign {
    Plus,
    Minus,
}
