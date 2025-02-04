use super::*;

impl Distance {
    pub fn floyd(&mut self) {
        // let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
        let mut graph: Distance = Distance::new(self.order);
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
    assert_eq!(graph.neighbors(0), vec![1, 2]);
    assert_eq!(graph.neighbors(1), vec![0, 2]);
    assert_eq!(graph.neighbors(2), vec![0, 1]);
    assert_eq!(graph.neighbors(3), vec![]);

    // Disconnect
    graph.disconnect([0, 1]);
    assert_eq!(graph.neighbors(0), vec![2]);
    assert_eq!(graph.neighbors(1), vec![2]);

    // Delete
    println!("graph: {graph}");
    graph.delete(1);
    println!("graph: {graph}");
    assert_eq!(graph.neighbors(0), vec![1]);
    assert_eq!(graph.neighbors(2), vec![]);
    assert_eq!(graph.neighbors(1), vec![0]);
}

#[test]
fn chordless_cycles() {
    let mut graph = Distance::new(4);
    // Connect
    graph.connect([0, 1]);
    graph.connect([1, 2]);
    graph.connect([2, 3]);

    println!("chordless_cycles:");
    graph.bfs_apsp();
    graph.connect([2, 0]);
}

#[test]
fn contract_edge() {
    let mut graph = Distance::tetrahedron();
    println!("tetrahedron: {graph}");
    println!("contracting [0, 2]......");
    graph.contract_edge([0, 2]);
    println!("contracted: {graph}");
    let mut triangle = Distance::new(3);
    triangle[[0, 1]] = 1;
    triangle[[1, 2]] = 1;
    triangle[[2, 0]] = 1;
    println!("expectation: {triangle}");
    assert_eq!(graph, triangle);
}

#[test]
fn bfs_apsp() {
    let mut distance = Distance::new(4);
    distance.connect([0, 1]);
    distance.connect([1, 2]);
    distance.connect([2, 3]);
    distance.bfs_apsp();
    assert_eq!(distance[[0, 2]], 2);
    assert_eq!(distance[[1, 3]], 2);
    assert_eq!(distance[[0, 3]], 3);

    /*
         *
    [
        [0, 1, -1, -1],
        [1, 0, 1, -1],
        [-1, 1, 0, 1],
        [-1, -1, 1, 0],
    ]
         */
}
