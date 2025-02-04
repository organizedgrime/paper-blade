use std::hash::Hash;
use std::ops::Index;

pub type VertexId = usize;

#[derive(Clone, Copy, Debug)]
pub struct Edge([VertexId; 2]);

impl Into<[VertexId; 2]> for Edge {
    fn into(self) -> [VertexId; 2] {
        self.0
    }
}

impl Into<Edge> for [VertexId; 2]  {
    fn into(self) -> Edge {
        Edge(self)
    }
}
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        (self[0] == other[0] &&
        self[1] == other[1]) ||
        (self[1] == other[0] &&
        self[0]== other[1] )
    }
}

impl Eq for Edge { }

impl Index<usize> for Edge {
    type Output = VertexId;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Hash for Edge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self[0] < self[1]  {
            self[0].hash(state);
            self[1].hash(state);
        } else {
            self[1].hash(state);
            self[0].hash(state);
        }
    }
}
