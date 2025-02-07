use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use crate::polyhedron::{shape::Distance, Edge, VertexId};

pub struct LRState {
    pub graph: Distance,
    pub roots: Vec<VertexId>,
    pub height: HashMap<VertexId, usize>,
    pub edge_parent: HashMap<VertexId, Edge>,
    pub low_point: HashMap<Edge, usize>,
    pub low_point_2: HashMap<Edge, usize>,
    pub low_point_edge: HashMap<Edge, Edge>,
    pub nesting_depth: HashMap<Edge, usize>,
    pub stack: Vec<ConflictPair<Edge>>,
    pub stack_emarker: HashMap<Edge, ConflictPair<Edge>>,
    pub eref: HashMap<Edge, Edge>,
    /// side of edge, or modifier for side of reference edge.
    pub side: HashMap<Edge, Sign>,
}

pub type Time = usize;

#[derive(Copy, Clone, Debug)]
pub enum DfsEvent {
    Discover(VertexId),
    /// An edge of the tree formed by the traversal.
    TreeEdge(Edge),
    /// An edge to an already visited node.
    BackEdge(Edge),
    /// A cross or forward edge.
    ///
    /// For an edge *(u, v)*, if the discover time of *v* is greater than *u*,
    /// then it is a forward edge, else a cross edge.
    CrossForwardEdge(Edge),
    /// All edges from a node have been reported.
    Finish(VertexId),
}

/// Similar to ``DfsEvent`` plus an extra event ``FinishEdge``
/// that indicates that we have finished processing an edge.
pub enum LRTestDfsEvent {
    Finish(VertexId),
    TreeEdge(Edge),
    BackEdge(Edge),
    FinishEdge(Edge),
}

pub struct NonPlanar {}

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
            DfsEvent::Discover(v) => {
                if let Entry::Vacant(entry) = self.height.entry(v) {
                    entry.insert(0);
                    self.roots.push(v);
                }
            }
            DfsEvent::TreeEdge(edge) => {
                let [v, w] = edge.inner();
                let v_height = self.height[&v];
                let w_height = v_height + 1;

                self.edge_parent.insert(w, edge);
                self.height.insert(w, w_height);
                // now initialize low points.
                self.low_point.insert(edge, v_height);
                self.low_point_2.insert(edge, w_height);
            }
            DfsEvent::BackEdge(edge) => {
                // do *not* consider ``(v, w)`` as a back edge if ``(w, v)`` is a tree edge.
                let [v, w] = edge.inner();
                if Some(&edge) != self.edge_parent.get(&v) {
                    self.low_point.insert(edge, self.height[&w]);
                    self.low_point_2.insert(edge, self.height[&v]);
                }
            }
            DfsEvent::Finish(v) => {
                for w in self.graph.neighbors(v) {
                    let edge: Edge = [v, w].into();

                    // determine nesting depth.
                    let low = match self.low_point.get(&edge) {
                        Some(val) => *val,
                        None =>
                        // if ``lowpt`` does *not* contain edge ``(v, w)``, it means
                        // that it's *not* a tree or a back edge so we skip it since
                        // it's oriented in the reverse direction.
                        {
                            continue
                        }
                    };

                    if self.low_point_2[&edge] < self.height[&v] {
                        // if it's chordal, add one.
                        self.nesting_depth.insert(edge, 2 * low + 1);
                    } else {
                        self.nesting_depth.insert(edge, 2 * low);
                    }

                    // update lowpoints of parent edge.
                    if let Some(e_par) = self.edge_parent.get(&v) {
                        match self.low_point[&edge].cmp(&self.low_point[e_par]) {
                            Ordering::Less => {
                                self.low_point_2.insert(
                                    *e_par,
                                    self.low_point[e_par].min(self.low_point_2[&edge]),
                                );
                                self.low_point.insert(*e_par, self.low_point[&edge]);
                            }
                            Ordering::Greater => {
                                modify_if_min(&mut self.low_point_2, *e_par, self.low_point[&edge]);
                            }
                            _ => {
                                let val = self.low_point_2[&edge];
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
            LRTestDfsEvent::TreeEdge(edge) => {
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(edge, last);
                }
            }
            LRTestDfsEvent::BackEdge(edge) => {
                if let Some(&last) = self.stack.last() {
                    self.stack_emarker.insert(edge, last);
                }
                self.low_point_edge.insert(edge, edge);
                let c_pair = ConflictPair::new(Interval::default(), Interval::new(edge, edge));
                self.stack.push(c_pair);
            }
            LRTestDfsEvent::FinishEdge(edge) => {
                let [v, w] = edge.inner();
                if self.low_point[&edge] < self.height[&v] {
                    // ei has return edge
                    let e_par = self.edge_parent[&v];
                    let val = self.low_point_edge[&edge];

                    match self.low_point_edge.entry(e_par) {
                        Entry::Occupied(_) => {
                            self.add_constraints(edge, e_par)?;
                        }
                        Entry::Vacant(o) => {
                            o.insert(val);
                        }
                    }
                }
            }
            LRTestDfsEvent::Finish(v) => {
                if let Some(&edge) = self.edge_parent.get(&v) {
                    let u = edge.v();
                    self.remove_back_edges(u);

                    // side of ``e = (u, v)` is side of a highest return edge
                    if self.low_point[&edge] < self.height[&u] {
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
                            self.eref.insert(edge, *e_high);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn until_top_of_stack_hits_emarker(&mut self, edge: Edge) -> Option<ConflictPair<Edge>> {
        if let Some(&c_pair) = self.stack.last() {
            if self.stack_emarker[&edge] != c_pair {
                return self.stack.pop();
            }
        }

        None
    }

    fn until_top_of_stack_is_conflicting(&mut self, ei: Edge) -> Option<ConflictPair<Edge>> {
        if let Some(c_pair) = self.stack.last() {
            if c_pair.left.conflict(self, ei) || c_pair.right.conflict(self, ei) {
                return self.stack.pop();
            }
        }

        None
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
            if self.low_point[qr_low] > self.low_point[&e] {
                // merge intervals
                self.union_intervals(&mut c_pair.right, q_pair.right);
            } else {
                // make consinsent
                self.eref.insert(*qr_low, self.low_point_edge[&e]);
            }
        }

        // merge conflicting return edges of e1, . . . , ei−1 into ``c_pair.left``.
        while let Some(mut q_pair) = self.until_top_of_stack_is_conflicting(ei) {
            if q_pair.right.conflict(self, ei) {
                q_pair.swap();

                if q_pair.right.conflict(self, ei) {
                    return Err(NonPlanar {});
                }
            }

            // merge interval below lowpt(ei) into ``c_pair.right``.
            if let Some((qr_low, qr_high)) = q_pair.right.as_ref() {
                if let Some(pr_low) = c_pair.right.as_mut_low() {
                    self.eref.insert(*pr_low, *qr_high);
                    *pr_low = *qr_low;
                }
            };
            self.union_intervals(&mut c_pair.left, q_pair.left);
        }

        if !c_pair.is_empty() {
            self.stack.push(c_pair);
        }

        Ok(())
    }

    /// Unify intervals ``pi``, ``qi``.
    ///
    /// Interval ``qi`` must be non - empty and contain edges
    /// with smaller lowpt than interval ``pi``.
    fn union_intervals(&mut self, pi: &mut Interval<Edge>, qi: Interval<Edge>) {
        match pi.as_mut_low() {
            Some(p_low) => {
                let (q_low, q_high) = qi.unwrap();
                self.eref.insert(*p_low, q_high);
                *p_low = q_low;
            }
            None => {
                *pi = qi;
            }
        }
    }
    // fn is_target<G: GraphBase>(edge: Option<&Edge<G>>, v: G::NodeId) -> Option<&Edge<G>> {
    //     edge.filter(|e| e.1 == v)
    // }
    fn follow_eref_until_is_target(&self, edge: Edge, v: VertexId) -> Option<Edge> {
        let mut res = Some(&edge);
        while let Some(b) = res.filter(|e| e.w() == v) {
            res = self.eref.get(b);
        }

        res.copied()
    }

    fn until_lowest_top_of_stack_has_height(&mut self, v: VertexId) -> Option<ConflictPair<Edge>> {
        if let Some(c_pair) = self.stack.last() {
            if c_pair.lowest(self) == self.height[&v] {
                return self.stack.pop();
            }
        }

        None
    }
    /// Trim back edges ending at parent v.
    fn remove_back_edges(&mut self, v: VertexId) {
        // drop entire conflict pairs.
        while let Some(c_pair) = self.until_lowest_top_of_stack_has_height(v) {
            if let Some(pl_low) = c_pair.left.low() {
                self.side.insert(*pl_low, Sign::Minus);
            }
        }

        // one more conflict pair to consider.
        if let Some(mut c_pair) = self.stack.pop() {
            // trim left interval.
            if let Some((pl_low, pl_high)) = c_pair.left.as_mut() {
                match self.follow_eref_until_is_target(*pl_high, v) {
                    Some(val) => {
                        *pl_high = val;
                    }
                    None => {
                        // just emptied.
                        // We call unwrap since right interval cannot be empty for otherwise
                        // the entire conflict pair had been removed.
                        let pr_low = c_pair.right.low().unwrap();
                        self.eref.insert(*pl_low, *pr_low);
                        self.side.insert(*pl_low, Sign::Minus);
                        c_pair.left = Interval::default();
                    }
                }
            }

            // trim right interval
            if let Some((pr_low, ref mut pr_high)) = c_pair.right.as_mut() {
                match self.follow_eref_until_is_target(*pr_high, v) {
                    Some(val) => {
                        *pr_high = val;
                    }
                    None => {
                        // just emptied.
                        // We call unwrap since left interval cannot be empty for otherwise
                        // the entire conflict pair had been removed.
                        let pl_low = c_pair.left.low().unwrap();
                        self.eref.insert(*pr_low, *pl_low);
                        self.side.insert(*pr_low, Sign::Minus);
                        c_pair.right = Interval::default();
                    }
                };
            }

            if !c_pair.is_empty() {
                self.stack.push(c_pair);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct ConflictPair<T> {
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

impl ConflictPair<Edge> {
    /// Returns the lowest low point of a conflict pair.
    fn lowest(&self, lr_state: &LRState) -> usize {
        match (self.left.low(), self.right.low()) {
            (Some(l_low), Some(r_low)) => lr_state.low_point[l_low].min(lr_state.low_point[r_low]),
            (Some(l_low), None) => lr_state.low_point[l_low],
            (None, Some(r_low)) => lr_state.low_point[r_low],
            (None, None) => usize::MAX,
        }
    }
}
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Interval<T> {
    inner: Option<(T, T)>,
}

impl<T> Default for Interval<T> {
    fn default() -> Self {
        Self { inner: None }
    }
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

impl Interval<Edge> {
    /// Returns ``true`` if the interval conflicts with ``edge``.
    fn conflict(&self, lr_state: &LRState, edge: Edge) -> bool {
        match self.inner {
            Some((_, ref h)) => lr_state.low_point.get(h) > lr_state.low_point.get(&edge),
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

pub enum Sign {
    Plus,
    Minus,
}
