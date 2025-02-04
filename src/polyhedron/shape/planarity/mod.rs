
use super::Distance;
mod state;
use state::LRState;

pub fn is_planar(graph: Distance) -> bool {
    let mut state = LRState::new(graph);

    // DFS orientation phase

    // L-R partition phase

    false
}

pub fn dfs_visitor(g: Distance) {

}
