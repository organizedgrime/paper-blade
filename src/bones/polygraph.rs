use crate::bones::*;
use rand::random;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::{collections::VecDeque, fmt::Display};
use ultraviolet::Vec3;
type VertMap<T> = HashMap<VertexId, T>;
pub type VertexId = usize;

#[derive(Debug, Clone, Default)]
pub struct PolyGraph {
    /// Conway Polyhedron Notation
    pub name: String,

    /// [Actual Graph]
    pub vertices: HashSet<VertexId>,
    /// Vertices that are adjacent
    pub edges: HashSet<Edge>,

    /// All Springs
    pub springs: HashSet<Edge>,

    /// [Derived Properties]
    /// Faces are simple cycles
    pub cycles: Vec<Face>,
    /// Distance matrix
    pub dist: HashMap<Edge, usize>,

    /// [Render Properties]
    /// Positions in 3D space
    pub positions: VertMap<Vec3>,
    /// Speeds
    pub speeds: VertMap<Vec3>,
    /// Edges in the process of contracting visually
    pub transactions: Vec<Transaction>,
    /// Edge length
    pub edge_length: f32,
    //pub contractions: HashSet<Edge>,
}

impl PolyGraph {
    /// New with n vertices
    pub fn new_disconnected(vertex_count: usize) -> Self {
        Self {
            vertices: (0..vertex_count).collect(),
            speeds: (0..vertex_count).map(|x| (x, Vec3::zero())).collect(),
            edge_length: 1.0,
            ..Default::default()
        }
    }

    // Use a Fibonacci Lattice to spread the points evenly around a sphere
    pub fn lattice(&mut self) {
        // Use a Fibonacci Lattice to evently distribute starting points on a sphere
        let phi = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        for (i, v) in self.vertices.iter().enumerate() {
            let y = 1.0 - (i as f32 / (self.vertices.len() - 1) as f32);
            let radius = (1.0 - y * y).sqrt();
            let theta = (phi * (i as f32)) % (std::f32::consts::PI * 2.0);
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            self.positions.insert(*v, Vec3::new(x, y, z));
        }
    }

    pub fn connect(&mut self, e: impl Into<Edge>) {
        let e = e.into();
        if e.v() != e.u() {
            self.edges.insert(e);
        }
    }

    pub fn disconnect(&mut self, e: impl Into<Edge>) {
        self.edges.remove(&e.into());
    }

    pub fn insert(&mut self) -> VertexId {
        let new_id = self.vertices.iter().max().unwrap() + 1;
        self.vertices.insert(new_id);
        // Position and speed
        self.positions
            .insert(new_id, Vec3::new(random(), random(), random()).normalized());
        self.speeds.insert(new_id, Vec3::zero());
        new_id
    }

    pub fn delete(&mut self, v: VertexId) {
        self.vertices.remove(&v);

        self.edges = self
            .edges
            .clone()
            .into_iter()
            .filter(|e| !e.contains(v))
            .collect();

        self.cycles = self
            .cycles
            .clone()
            .into_iter()
            .map(|face| face.into_iter().filter(|&u| u != v).collect())
            .collect();

        self.positions.remove(&v);
        self.speeds.remove(&v);
    }

    /// Edges of a vertex
    pub fn edges(&self, v: VertexId) -> Vec<Edge> {
        self.edges
            .iter()
            .filter_map(|e| if e.other(v).is_some() { Some(*e) } else { None })
            .collect()
    }

    /// Number of faces
    pub fn face_count(&self) -> i64 {
        2 + self.edges.len() as i64 - self.vertices.len() as i64
    }

    // Vertices that are connected to a given vertex
    pub fn connections(&self, v: VertexId) -> HashSet<VertexId> {
        self.edges.iter().filter_map(|e| e.other(v)).collect()
    }

    /// All faces
    pub fn find_cycles(&mut self) {
        let mut triplets = Vec::<Face>::new();
        let mut cycles = HashSet::<Face>::default();

        // find all the triplets
        for &u in self.vertices.iter() {
            let adj: HashSet<VertexId> = self.connections(u);
            for &x in adj.iter() {
                for &y in adj.iter() {
                    if x != y && u < x && x < y {
                        let new_face = Face::new(vec![x, u, y]);
                        if self.edges.contains(&(x, y).into()) {
                            cycles.insert(new_face);
                        } else {
                            triplets.push(new_face);
                        }
                    }
                }
            }
        }
        // while there are unparsed triplets
        while !triplets.is_empty() && (cycles.len() as i64) < self.face_count() {
            let p = triplets.remove(0);
            // for each v adjacent to u_t
            for v in self.connections(p[p.len() - 1]) {
                if v > p[1] {
                    let c = self.connections(v);
                    // if v is not a neighbor of u_2..u_t-1
                    if !p[1..p.len() - 1].iter().any(|vi| c.contains(vi)) {
                        let mut new_face = p.clone();
                        new_face.push(v);
                        if self.connections(p[0]).contains(&v) {
                            //cycles.remo
                            cycles.insert(new_face);
                        } else {
                            triplets.push(new_face);
                        }
                    }
                }
            }
        }

        self.cycles = cycles.into_iter().collect();
    }

    pub fn pst(&mut self) {
        if self.edges.is_empty() {
            return;
        }

        let n = self.vertices.len();
        // Vertex
        //
        // d-queues associated w each vertex
        // maps from v -> ( maps from d -> u )
        let mut dqueue: HashMap<VertexId, VecDeque<(VertexId, usize)>> = Default::default();
        //
        let mut children: HashMap<VertexId, Vec<VertexId>> = Default::default();

        // Counters for vertices whos shortest paths have already been obtained
        let mut counters: HashMap<VertexId, usize> =
            self.vertices.iter().map(|v| (*v, n - 1)).collect();

        // The element D[i, j] represents the distance from v_i to vj.
        let mut dist: HashMap<Edge, usize> = Default::default();

        // d = 0
        let mut depth = 1;
        // while 0 < |V|
        loop {
            let verts: HashSet<VertexId> = counters
                .iter()
                .filter_map(|(v, c)| if *c == 0 { None } else { Some(*v) })
                .collect();
            if verts.is_empty() {
                break;
            }

            let mut removed = false;

            for v in verts.into_iter() {
                // for v in V
                // START EXTEND(v, d, D, S)
                if depth == 1 {
                    //
                    for e in self.edges(v) {
                        // Connected node
                        let w = e.other(v).unwrap();
                        // D[w.id, v.id] = d
                        dist.insert(e, 1);
                        // add w' to v'.children
                        children.entry(v).or_default().push(w);
                        // v.que.enque(w', 1)
                        dqueue.entry(v).or_default().push_back((w, 1));
                        dqueue.entry(w).or_default().push_back((v, 1));
                        // v.c = v.c + 1
                        *counters.entry(v).or_default() -= 1;
                        //*counters.entry(w).or_default() -= 1;
                        removed = true;
                    }
                } else {
                    // w = v.que.deque(d - 1)
                    // while w is not None:
                    'dq: loop {
                        let vqueue = dqueue.entry(v).or_default();
                        if let Some((w, d)) = vqueue.pop_front() {
                            if d != depth - 1 {
                                dqueue.entry(v).or_default().push_back((w, d));
                                break;
                            }
                            // for x in w.children
                            for x in children.entry(w).or_default().clone() {
                                let e: Edge = (x, v).into();
                                if x != v && !dist.contains_key(&e) {
                                    // D[x.id, v.id] = d;
                                    dist.insert(e, depth);
                                    // add x' to w' children
                                    children.entry(w).or_default().push(x);
                                    // v.que.enque(x', d)
                                    dqueue.entry(v).or_default().push_back((x, depth));
                                    dqueue.entry(x).or_default().push_back((v, depth));
                                    // v.c = v.c + 1
                                    removed = true;
                                    *counters.entry(v).or_default() -= 1;
                                    *counters.entry(x).or_default() -= 1;
                                    // if v.c == n: return
                                    if *counters.entry(x).or_default() == 0
                                        && *counters.entry(w).or_default() == 0
                                        && *counters.entry(v).or_default() == 0
                                    {
                                        break 'dq;
                                    }
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            // END EXTEND
            // d = d + 1
            depth += 1;

            if !removed {
                self.dist = dist;
                log::error!("failed distance computation");
                return;
            }
        }

        self.dist = dist;
    }

    pub fn springs(&mut self) {
        let diameter = { *self.dist.values().max().unwrap_or(&1) };
        self.springs = self
            .vertices
            .iter()
            .flat_map(|&v| self.vertices.iter().map(move |&u| Edge::from((v, u))))
            .filter(|e| e.u() != e.v())
            .filter(|e| self.dist[e] <= 2 || self.dist[e] >= diameter - 1)
            .collect::<HashSet<_>>();

        log::debug!(
            "v_len: {} | v2: {} | springs: {}",
            self.vertices.len(),
            (self.vertices.len() as f32).powi(2),
            self.springs.len()
        );
    }
}

impl Display for PolyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut vertices = self.vertices.iter().collect::<Vec<_>>();
        vertices.sort();
        let mut adjacents = self.edges.clone().into_iter().collect::<Vec<_>>();
        adjacents.sort();

        f.write_fmt(format_args!(
            "name:\t\t{}\nvertices:\t{:?}\nadjacents:\t{}\nfaces:\t\t{}\n",
            self.name,
            vertices,
            adjacents
                .iter()
                .fold(String::new(), |acc, e| format!("{e}, {acc}")),
            self.cycles.iter().fold(String::new(), |acc, f| format!(
                "[{}], {acc}",
                f.iter().fold(String::new(), |acc, x| format!("{x}, {acc}"))
            ))
        ))
    }
}

#[cfg(test)]
impl PolyGraph {
    pub fn floyd(&mut self) {
        // let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
        let mut dist: HashMap<VertexId, HashMap<VertexId, u32>> = self
            .vertices
            .iter()
            .map(|v| {
                (
                    *v,
                    self.vertices
                        .iter()
                        .map(|u| {
                            (
                                *u,
                                if u == v {
                                    0
                                } else if self.edges.contains(&(v, u).into()) {
                                    1
                                } else {
                                    u32::MAX
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        for k in self.vertices.iter() {
            for i in self.vertices.iter() {
                for j in self.vertices.iter() {
                    if dist[i][k] != u32::MAX && dist[k][j] != u32::MAX {
                        let nv = dist[i][k] + dist[k][j];
                        if dist[i][j] > nv || dist[j][i] > nv {
                            dist.entry(*i).or_default().insert(*j, nv);
                            dist.entry(*j).or_default().insert(*i, nv);
                        }
                    }
                }
            }
        }

        let mut dd = HashMap::default();
        for v in self.vertices.iter() {
            for u in self.vertices.iter() {
                let dvu = dist[v][u];
                if dvu != u32::MAX && dvu != 0 {
                    let e = (v, u).into();
                    dd.insert(e, dvu as usize);
                }
            }
        }

        self.dist = dd;
    }
}

#[cfg(test)]
mod test {
    use crate::bones::{Face, PolyGraph};
    use std::collections::HashSet;
    use test_case::test_case;

    #[test_case(PolyGraph::pyramid(3); "T")]
    #[test_case(PolyGraph::prism(4); "C")]
    #[test_case(PolyGraph::octahedron(); "O")]
    #[test_case(PolyGraph::dodecahedron(); "D")]
    #[test_case(PolyGraph::icosahedron(); "I")]
    #[test_case({ let mut g = PolyGraph::prism(4); g.truncate(None); g.pst(); g} ; "tC")]
    #[test_case({ let mut g = PolyGraph::octahedron(); g.truncate(None); g.pst(); g} ; "tO")]
    #[test_case({ let mut g = PolyGraph::dodecahedron(); g.truncate(None); g.pst(); g} ; "tD")]
    fn pst(mut graph: PolyGraph) {
        let new_dist = graph.dist.clone();
        graph.dist = Default::default();
        graph.floyd();
        let old_dist = graph.dist.clone();

        //assert_eq!(old_dist, graph.dist);
        assert_eq!(
            old_dist
                .clone()
                .into_keys()
                .collect::<HashSet<_>>()
                .difference(&new_dist.clone().into_keys().collect::<HashSet<_>>())
                .collect::<HashSet<_>>(),
            HashSet::new()
        );

        let o1 = old_dist
            .clone()
            .into_iter()
            .map(|(k, v)| (k.id().0, k.id().1, v))
            .collect::<HashSet<_>>();
        let o2 = &new_dist
            .clone()
            .into_iter()
            .map(|(k, v)| (k.id().0, k.id().1, v))
            .collect::<HashSet<_>>();

        assert_eq!(
            o1.difference(o2).collect::<HashSet<_>>(),
            o2.difference(&o1).collect::<HashSet<_>>()
        );
        assert_eq!(old_dist, new_dist);
    }

    #[test]
    fn basics() {
        let mut graph = PolyGraph::new_disconnected(4);
        // Connect
        graph.connect((0, 1));
        graph.connect((0, 2));
        graph.connect((1, 2));
        assert_eq!(graph.connections(0), vec![1, 2].into_iter().collect());
        assert_eq!(graph.connections(1), vec![0, 2].into_iter().collect());
        assert_eq!(graph.connections(2), vec![0, 1].into_iter().collect());
        assert_eq!(graph.connections(3), vec![].into_iter().collect());

        // Disconnect
        graph.disconnect((0, 1));
        assert_eq!(graph.connections(0), vec![2].into_iter().collect());
        assert_eq!(graph.connections(1), vec![2].into_iter().collect());

        // Delete
        graph.delete(1);
        assert_eq!(graph.connections(0), vec![2].into_iter().collect());
        assert_eq!(graph.connections(1), vec![].into_iter().collect());
        assert_eq!(graph.connections(2), vec![0].into_iter().collect());
    }

    #[test]
    fn chordless_cycles() {
        let mut graph = PolyGraph::new_disconnected(4);
        // Connect
        graph.connect((0, 1));
        graph.connect((1, 2));
        graph.connect((2, 3));

        graph.pst();
        assert_eq!(graph.cycles.len(), 0);

        graph.connect((2, 0));
        graph.pst();
        graph.find_cycles();
        assert_eq!(graph.cycles, vec![Face::new(vec![0, 1, 2])]);
    }
}
