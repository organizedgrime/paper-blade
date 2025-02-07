use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use super::{Distance, Edge, VertexId};
mod state;
use state::{DfsEvent, LRState, LRTestDfsEvent, NonPlanar, Time};

impl Distance {
    pub fn is_planar(&self) -> bool {
        let state = &mut LRState::new(self);

        let time = &mut 0;
        let discovered = &mut HashSet::with_capacity(self.order());
        let finished = &mut HashSet::with_capacity(self.order());

        // DFS orientation phase
        for node in self.vertices() {
            dfs_visitor(self.clone(), node, state, discovered, finished, time);
        }

        // L-R partition phase

        for v in state.roots.clone() {
            // let res = lr_visit_ordered_dfs_tree(&mut state, v, |state, event| {
            //     state.lr_testing_visitor(event)
            // });
            // if res.is_err() {
            //     return false;
            // }
        }

        false
    }
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
) -> Control<usize> {
    if !discovered.insert(u) {
        return Control::Continue;
    }
    state.lr_orientation_visitor(DfsEvent::Discover(u, time_post_inc(time)));

    let mut stack: Vec<(VertexId, Vec<VertexId>)> = Vec::new();
    stack.push((u, graph.neighbors(u)));

    while let Some((u, neighbors)) = stack.last() {
        let mut next = None;
        for &v in neighbors {
            // is_visited
            if !discovered.contains(&v) {
                //try_control!(visitor(), continue);
                state.lr_orientation_visitor(DfsEvent::TreeEdge(*u, v));
                discovered.insert(v);
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::Discover(v, time_post_inc(time)));
                //     continue
                // );
                next = Some(v);
                break;
            } else if !finished.contains(&v) {
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::BackEdge(*u, v));
                // , continue);
            } else {
                // try_control!(
                state.lr_orientation_visitor(DfsEvent::CrossForwardEdge(*u, v));
                //     continue
                // );
            }
        }

        match next {
            Some(v) => stack.push((v, graph.neighbors(v))),
            None => {
                let first_finish = finished.insert(*u);
                debug_assert!(first_finish);
                state.lr_orientation_visitor(DfsEvent::Finish(*u, time_post_inc(time)));
                //panic!("Pruning on the `DfsEvent::Finish` is not supported!")
                stack.pop();
            }
        };
    }

    Control::Continue
}

fn time_post_inc(x: &mut Time) -> Time {
    let v = *x;
    *x += 1;
    v
}
// Filter edges by key and sort by nesting depth
// This allows us to ignore edges which are not tree or back edges,
// meaning we can skip it because it's going the wrong direction.
fn remaining_edges(w: VertexId, lr_state: &LRState) -> Vec<Edge> {
    let mut edges: Vec<Edge> = lr_state
        .graph
        .neighbors(w)
        .into_iter()
        .filter_map(|v| {
            let e: Edge = [v, w].into();
            lr_state.low_point.contains_key(&e).then_some(e)
        })
        .collect();
    edges.sort_by_key(|edge| lr_state.nesting_depth[edge]);
    // Remove parallel edges, which have no impact on planarity
    edges.dedup();
    edges
}

/// Visits the DFS - oriented tree that we have pre-computed
/// and stored in ``lr_state``. We traverse the edges of
/// a node in nesting depth order. Events are emitted at points
/// of interest and should be handled by ``visitor``.
fn lr_visit_ordered_dfs_tree(lr_state: &mut LRState, v: VertexId) -> Result<(), NonPlanar> {
    let mut stack: Vec<(VertexId, Vec<Edge>)> = vec![(v, remaining_edges(v, lr_state))];

    while let Some(elem) = stack.last_mut() {
        let v = elem.0;
        let adjacent_edges = elem.1.clone();
        let mut next = None;

        {
            for edge in adjacent_edges {
                if Some(edge) == lr_state.edge_parent.get(&edge.w()).copied() {
                    lr_state.lr_testing_visitor(LRTestDfsEvent::TreeEdge(edge))?;
                    next = Some(edge.w());
                    break;
                } else {
                    lr_state.lr_testing_visitor(LRTestDfsEvent::BackEdge(edge))?;
                    lr_state.lr_testing_visitor(LRTestDfsEvent::FinishEdge(edge))?;
                }
            }
        }

        match next {
            Some(w) => {
                stack.push((w, remaining_edges(w, lr_state)));
            }
            None => {
                stack.pop();
                lr_state.lr_testing_visitor(LRTestDfsEvent::Finish(v))?;

                if let Some(&edge) = lr_state.edge_parent.get(&v) {
                    lr_state.lr_testing_visitor(LRTestDfsEvent::FinishEdge(edge))?;
                }
            }
        }
    }

    Ok(())
}
