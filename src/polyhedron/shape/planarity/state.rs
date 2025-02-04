use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use crate::polyhedron::{shape::Distance, Edge, VertexId};

pub struct LRState {
    graph: Distance,
    pub roots: Vec<VertexId>,
    height: HashMap<VertexId, usize>,
    edge_parent: HashMap<VertexId, Edge>,
    low_point: HashMap<Edge, usize>,
    low_point_2: HashMap<Edge, usize>,
    low_point_edge: HashMap<Edge, Edge>,
    nesting_depth: HashMap<Edge, usize>,
    stack: Vec<ConflictPair<Edge>>,
    stack_emarker: HashMap<Edge, ConflictPair<Edge>>,
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

/// Similar to ``DfsEvent`` plus an extra event ``FinishEdge``
/// that indicates that we have finished processing an edge.
enum LRTestDfsEvent {
    Finish(VertexId),
    TreeEdge(VertexId, VertexId),
    BackEdge(VertexId, VertexId),
    FinishEdge(VertexId, VertexId),
}

struct NonPlanar {}

impl LRState {
    pub fn new(graph: &Distance) -> Self {
        let e = graph.edges().count();
        Self {
            graph: graph.clone(),
            roots: Vec::new(),
            height: HashMap::with_capacity(graph.order()),
            edge_parent: HashMap::with_capacity(e),
            low_point: HashMap::with_capacity(e),
            low_point_2: HashMap::with_capacity(e),
            low_point_edge: HashMap::with_capacity(e),
            nesting_depth: HashMap::with_capacity(e),
            stack: Vec::new(),
            stack_emarker: HashMap::with_capacity(e),
            eref: HashMap::with_capacity(e),
            side: graph.edges().map(|e| (e.into(), Sign::Plus)).collect(),
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

    pub fn lr_testing_visitor(&mut self, event: LRTestDfsEvent) -> Result<(), NonPlanar> {
        match event {
            LRTestDfsEvent::TreeEdge(v, w) => {
                let ei = (v, w);
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(ei, last);
                }
            }
            LRTestDfsEvent::BackEdge(v, w) => {
                let ei = (v, w);
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(ei, last);
                }
                self.lowpt_edge.insert(ei, ei);
                let c_pair = ConflictPair::new(Interval::default(), Interval::new(ei, ei));
                self.stack.push(c_pair);
            }
            LRTestDfsEvent::FinishEdge(v, w) => {
                let ei = (v, w);
                if self.lowpt[&ei] < self.height[&v] {
                    // ei has return edge
                    let e_par = self.eparent[&v];
                    let val = self.lowpt_edge[&ei];

                    match self.lowpt_edge.entry(e_par) {
                        Entry::Occupied(_) => {
                            self.add_constraints(ei, e_par)?;
                        }
                        Entry::Vacant(o) => {
                            o.insert(val);
                        }
                    }
                }
            }
            LRTestDfsEvent::Finish(v) => {
                if let Some(&e) = self.eparent.get(&v) {
                    let u = e.0;
                    self.remove_back_edges(u);

                    // side of ``e = (u, v)` is side of a highest return edge
                    if self.low_point[&e] < self.height[&u] {
                        if let Some(top) = self.stack.last() {
                            let e_high = match (top.left.high(), top.right.high()) {
                                (Some(hl), Some(hr)) => {
                                    if self.low_point[hl] > self.low_point[hr] {
                                        hl
                                    } else {
                                        hr
                                    }
                                }
                                (Some(hl), None) => hl,
                                (None, Some(hr)) => hr,
                                _ => {
                                    // Otherwise ``top`` would be empty, but we don't push
                                    // empty conflict pairs in stack.
                                    unreachable!()
                                }
                            };
                            self.eref.insert(e, *e_high);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Adding constraints associated with edge ``ei``.
    fn add_constraints(&mut self, ei: Edge, e: Edge) -> Result<(), NonPlanar> {
        let mut c_pair = ConflictPair::<Edge>::default();

        // merge return edges of ei into ``c_pair.right``.
        while let Some(mut q_pair) = self.until_top_of_stack_hits_emarker(ei) {
            if !q_pair.left.is_empty() {
                q_pair.swap();

                if !q_pair.left.is_empty() {
                    return Err(NonPlanar {});
                }
            }

            // We call unwrap since ``q_pair`` was in stack and
            // ``q_pair.right``, ``q_pair.left`` can't be both empty
            // since we don't push empty conflict pairs in stack.
            let qr_low = q_pair.right.low().unwrap();
            if self.lowpt[qr_low] > self.lowpt[&e] {
                // merge intervals
                self.union_intervals(&mut c_pair.right, q_pair.right);
            } else {
                // make consinsent
                self.eref.insert(*qr_low, self.lowpt_edge[&e]);
            }
        }
    }

}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct ConflictPair<T> {
    left: Interval<T>,
    right: Interval<T>,
}

impl<T> Default for ConflictPair<T> {
    fn default() -> Self {
        ConflictPair {
            left: Interval::default(),
            right: Interval::default(),
        }
    }
}

impl<T> ConflictPair<T> {
    fn new(left: Interval<T>, right: Interval<T>) -> Self {
        ConflictPair { left, right }
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right)
    }

    fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }
}

impl<T> ConflictPair<(T, T)> {
    /// Returns the lowest low point of a conflict pair.
    fn lowest(&self, lr_state: &LRState) -> usize
    {
        match (self.left.low(), self.right.low()) {
            (Some(l_low), Some(r_low)) => lr_state.low_point[l_low].min(lr_state.low_point[r_low]),
            (Some(l_low), None) => lr_state.low_point[l_low],
            (None, Some(r_low)) => lr_state.low_point[r_low],
            (None, None) => usize::MAX,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
struct Interval<T> {
    inner: Option<(T, T)>,
}
impl<T> Interval<T> {
    fn new(low: T, high: T) -> Self {
        Interval {
            inner: Some((low, high)),
        }
    }

    fn is_empty(&self) -> bool {
        self.inner.is_none()
    }

    fn unwrap(self) -> (T, T) {
        self.inner.unwrap()
    }

    fn low(&self) -> Option<&T> {
        match self.inner {
            Some((ref low, _)) => Some(low),
            None => None,
        }
    }

    fn high(&self) -> Option<&T> {
        match self.inner {
            Some((_, ref high)) => Some(high),
            None => None,
        }
    }

    fn as_ref(&mut self) -> Option<&(T, T)> {
        self.inner.as_ref()
    }

    fn as_mut(&mut self) -> Option<&mut (T, T)> {
        self.inner.as_mut()
    }

    fn as_mut_low(&mut self) -> Option<&mut T> {
        match self.inner {
            Some((ref mut low, _)) => Some(low),
            None => None,
        }
    }
}

impl<T> Interval<(T, T)>
where
    T: Copy + Hash + Eq,
{
    /// Returns ``true`` if the interval conflicts with ``edge``.
    fn conflict<G>(&self, lr_state: &LRState<G>, edge: Edge<G>) -> bool
    where
        G: GraphBase<NodeId = T>,
    {
        match self.inner {
            Some((_, ref h)) => lr_state.lowpt.get(h) > lr_state.lowpt.get(&edge),
            _ => false,
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
