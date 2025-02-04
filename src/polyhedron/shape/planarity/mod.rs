
use std::collections::{HashMap, HashSet};

use super::{Distance, VertexId};
mod state;
use state::{DfsEvent, LRState, Time};

pub fn is_planar(graph: Distance) -> bool {
    let state = &mut LRState::new(graph.clone());

    let time = &mut 0;
    let discovered = &mut HashSet::with_capacity(graph.order());
    let finished = &mut HashSet::with_capacity(graph.order());

    // DFS orientation phase
    for node in graph.vertices() {
        dfs_visitor(graph.clone(), node, state, discovered, finished, time);
    }

    // L-R partition phase

    false
}

/// Control flow for `depth_first_search` callbacks.
#[derive(Copy, Clone, Debug)]
pub enum Control<B> {
    /// Continue the DFS traversal as normal.
    Continue,
    /// Prune the current node from the DFS traversal. No more edges from this
    /// node will be reported to the callback. A `DfsEvent::Finish` for this
    /// node will still be reported. This can be returned in response to any
    /// `DfsEvent`, except `Finish`, which will panic.
    Prune,
    /// Stop the DFS traversal and return the provided value.
    Break(B),
}

fn dfs_visitor(
    graph: Distance,
    u: VertexId,
    state: &mut LRState,
    //visitor: &mut F,
    //state.lr_orientation_visitor(event)
    discovered: &mut HashSet<VertexId>,
    finished: &mut HashSet<VertexId>,
    time: &mut Time,
) -> Control<usize>
{
    if !discovered.insert(u) {
        return Control::Continue;
    }
    state.lr_orientation_visitor(DfsEvent::Discover(u, time_post_inc(time)));
    Control::Continue

        /* let mut stack: Vec<(G::NodeId, <G as IntoEdges>::Edges)> = Vec::new();
        stack.push((u, graph.edges(u)));

        while let Some(elem) = stack.last_mut() {
            let u = elem.0;
            let adjacent_edges = &mut elem.1;
            let mut next = None;

            for edge in adjacent_edges {
                let v = edge.target();
                if !discovered.is_visited(&v) {
                    try_control!(visitor(DfsEvent::TreeEdge(u, v, edge.weight())), continue);
                    discovered.visit(v);
                    try_control!(
                        visitor(DfsEvent::Discover(v, time_post_inc(time))),
                        continue
                    );
                    next = Some(v);
                    break;
                } else if !finished.is_visited(&v) {
                    try_control!(visitor(DfsEvent::BackEdge(u, v, edge.weight())), continue);
                } else {
                    try_control!(
                        visitor(DfsEvent::CrossForwardEdge(u, v, edge.weight())),
                        continue
                    );
                }
            }

            match next {
                Some(v) => stack.push((v, graph.edges(v))),
                None => {
                    let first_finish = finished.visit(u);
                    debug_assert!(first_finish);
                    try_control!(
                        visitor(DfsEvent::Finish(u, time_post_inc(time))),
                        panic!("Pruning on the `DfsEvent::Finish` is not supported!")
                    );
                    stack.pop();
                }
            };
        } */
}
fn time_post_inc(x: &mut Time) -> Time {
    let v = *x;
    *x += 1;
    v
}
