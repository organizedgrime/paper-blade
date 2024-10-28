use super::*;
/*
    T = Y3
    O = aT (ambo tetrahedron)
    C = jT (join tetrahedron)
    I = sT (snub tetrahedron)
    D = gT (gyro tetrahedron)
*/

// Platonic Solids
impl JagGraph {
    pub fn prism(n: usize) -> JagGraph {
        let mut graph = JagGraph::new(n * 2);
        //p.name = format!("P{n}");
        for i in 0..n {
            // Lower polygon
            graph.connect([i % n, (i + 1) % n]);
            // Upper polygon
            graph.connect([(i % n) + n, ((i + 1) % n) + n]);
            // Connect
            graph.connect([(i % n), (i % n) + n]);
            graph.connect([(i + 1) % n, ((i + 1) % n) + n]);
        }
        graph.pst();
        graph.find_cycles();
        graph
    }

    pub fn anti_prism(n: usize) -> JagGraph {
        let mut graph = JagGraph::new(n * 2);
        //p.name = format!("A{n}");
        for i in 0..n {
            // Lower polygon
            graph.connect([i % n, (i + 1) % n]);
            // Upper polygon
            graph.connect([(i % n) + n, ((i + 1) % n) + n]);
            // Connect
            graph.connect([(i % n), (i % n) + n]);
            graph.connect([(i + 1) % n, ((i + 1) % n) + n]);

            graph.connect([(i % n), ((i + 1) % n) + n]);
        }
        graph.pst();
        graph.find_cycles();
        graph
    }

    pub fn pyramid(n: usize) -> JagGraph {
        let mut graph = JagGraph::new(n + 1);
        //graph.name = format!("Y{n}");
        for i in 0..n {
            graph.connect([i, (i + 1) % n]);
            graph.connect([i, n]);
        }
        graph.pst();
        graph.find_cycles();
        graph
    }

    pub fn octahedron() -> JagGraph {
        let mut p = JagGraph::pyramid(3);
        println!("P3:\n{p}");
        let edges = p.ambo();
        println!("aP3:\n{p}");
        //p.contract_edge(edges[0]);
        p.contract_edges(edges);
        println!("caP3:\n{p}");
        p.pst();
        println!("PSTcaP3:\n{p}");
        //p.springs();
        //p.lattice();
        //p.name = "O".into();
        p
    }

    pub fn dodecahedron() -> JagGraph {
        let mut graph = JagGraph::anti_prism(5);
        let edges = graph.expand(false);
        graph.contract_edges(edges);
        graph.truncate(Some(5));
        //p.pst();
        //p.springs();
        // graph.name = "D".into();
        graph
    }

    pub fn icosahedron() -> JagGraph {
        let mut graph = JagGraph::anti_prism(5);
        graph.kis(Some(5));
        //p.pst();
        //p.springs();
        //p.name = "I".into();
        graph
    }
}
